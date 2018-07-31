
require('utility.init')

local Supervised = torch.class('learner.Supervised')

function Supervised:__init(model, criterion, option)
  self.model = model
  self.criterion = criterion
  self.option = option
  self.network = self.model.network
  self.params, self.gradParams = utility.net.getParameters(self.network)
  if self.option.optim.multiLR == true then
    self.paramsTbl, self.gradParamsTbl = self.network:parameters()
  else
    self.paramsTbl, self.gradParamsTbl = {self.params}, {self.gradParams}
  end
  
  if self.option.lrFinder == false and self.option.optim.regime == nil then
    self.lrScheduler = utility.net.lrSchemes(option.optim)
  end
  
  -- create data cache type according to model type
  self.input = CUDA(torch[self.option.data.inputType]())
  self.target = CUDA(torch[self.option.data.targetType]())
end

function Supervised:modeFlag(set)
  if set == 'train' then
    self.network:training()
  elseif set == 'test' then
    self.network:evaluate()
  end
end

function Supervised:resetState(curEpoch, curBatchSz)
  self.latest['epoch'] = curEpoch
  self.latest['batchSz'] = curBatchSz
end

function Supervised:copyBatch(batchData)
  self.input:resize(batchData.input:size()):copy(batchData.input)
  self.target:resize(batchData.target:size()):copy(batchData.target)
end

function Supervised:forward(set)
  self.network:forward(self.input)
  self.loss['iter'] = self.criterion:forward(self.network.output, self.target)
  self:statisticsUpdate(set)
end

function Supervised:backward()
  self.criterion:backward(self.network.output, self.target)
  self.network:backward(self.input, self.criterion.gradInput)
end

function Supervised:update()
  self.network:zeroGradParameters()
  self:forward('train')
  self:backward()
  local loss = self.loss['iter']
  if torch.type(self.loss['iter']) == 'table' then -- when work with sequence
    loss = self.loss['iter'][0]
  end

  -- deal with learning rate
  if self.option.lrFinder == true then
    self.option.optim.state[1].learningRate = self.option.optim.lrMin *
      math.exp(-self.lrFinderStep * (self.latest['trainIter'] - 1))
    self.lrFinderRecord[1][self.latest['trainIter']] = self.option.optim.state[1].learningRate
    self.lrFinderRecord[2][self.latest['trainIter']] = self.loss['iter']
  else
    if self.option.optim.regime == nil then
      self.option.optim.state[1].learningRate = self.lrScheduler(self.latest['trainIter'])
    else
      for i = 1, #self.option.optim.regime.step do
        if self.option.optim.iEpoch >= self.option.optim.regime.step[i].begEpoch
        and self.option.optim.iEpoch <= self.option.optim.regime.step[i].endEpoch then
          if self.option.optim.multiLR == true then
            for j = 1, #self.option.optim.regime.step[i].group do
              for k, v in pairs(self.option.optim.regime.step[i].group[j].index) do
                self.option.optim.state[v] = utility.tbl.deepClone(self.option.optim.regime.step[i].group[j].state)
              end
            end
          else
            self.option.optim.state[1] = self.option.optim.regime.step[i].state
          end
        end
      end
    end
  end

  for i = 1, #self.option.optim.state do
    local function propagate(params)
      if params ~= self.paramsTbl[i] then -- double check parames is the same
        self.paramsTbl[i]:copy(params)
        error('maybe wrong??')
      end
      return loss, self.gradParamsTbl[i]
    end
    optim[self.option.optim.method](propagate, self.paramsTbl[i], self.option.optim.state[i])
  end

  assert(self.params:storage() == self.network:parameters()[1]:storage()) -- double-check getParameters right
end

function Supervised:iterPrint(set, epoch, iBatch, nBatch, dataTime, iterTime)
end

function Supervised:statistics(set, loader)
  if self.timer == nil then
    self.timer = {['data'] = torch.Timer(), ['iter'] = torch.Timer()} -- timer for each iteration and loading data
  end
  if self.loss == nil then
    self.loss = {['train'] = 0, ['test'] = 0, ['iter'] = 0}
  end
  if self.pastalog == nil and self.option.url ~= nil then
    self.pastalog = require('pastalog')
  end
  if self.latest == nil then
    self.latest = {['epoch'] = 0, ['batchSz'] = 0, ['trainIter'] = 0}
  end
  if self.sampleSz == nil then
    self.sampleSz = {}
  end
  if self.sampleSz[set] == nil then
    self.sampleSz[set] = loader:sampleSize()
  end
  if self.lrFinderStep == nil and self.option.lrFinder == true then
    -- lr = lrMin * exp(-step  * (iterIdx - 1))
    self.lrFinderStep = -math.log(self.option.optim.lrMax / self.option.optim.lrMin) /
      (self.option.optim.nTrainBatch * self.option.optim.maxEpoch - 1)
    self.lrFinderRecord = torch.Tensor(2, self.option.optim.nTrainBatch * self.option.optim.maxEpoch)
  end
  self.timer['iter']:reset()    -- reset iteration timer
  self.timer['data']:reset()    -- reset data processing timer
  self.loss[set] = 0            -- reset loss
end

function Supervised:statisticsUpdate(set)
  self.loss[set] = self.loss[set] + self.loss['iter'] * self.latest['batchSz']
end

-- function Supervised:summary(epoch)
--   for i,set in pairs({'train', 'test'}) do 
--     self.loss[set] = self.loss[set] / self.sampleSz[set]
--   end
--   if self.pastalog ~= nil then
--     local _, base, _ = utility.io.splitPath(self.option.log)
--     self.pastalog(base, 'trainLoss', self.loss['train'], epoch, self.option.url)
--     self.pastalog(base, 'validLoss', self.loss['test'], epoch, self.option.url)
--   end
-- end

function Supervised:summary(set, epoch)
  self.loss[set] = self.loss[set] / self.sampleSz[set]
  if self.pastalog ~= nil then
    local _, base, _ = utility.io.splitPath(self.option.log)
    self.pastalog(base, set .. 'Loss', self.loss[set], epoch, self.option.url)
  end
end

function Supervised:example(set, loader)
  if self.option.example.set ~= nil then
    self:modeFlag('test')
  end
end

function Supervised:measure()
  return self.loss['test']
end

function Supervised:train(epoch, loader)
  print('\n=> Training epoch # ' .. epoch)
  self.option.optim.iEpoch = epoch
  
  local set = 'train'
  self:modeFlag(set)                     -- set train flag for BN, dropout
  self:statistics(set, loader) -- reset whatever statistics for train phase in records
  
  for iBatch, batchData in loader:iterator() do
    self.latest['trainIter'] = (epoch - 1) * self.option.optim.nTrainBatch + iBatch
    self:copyBatch(batchData)               -- inline copy batch to GPU if use
    local dataTime = self.timer['data']:time().real

    self:resetState(epoch, batchData.size)  -- save current epoch and batch size, reset used for sequence models
    self:update()
    local iterTime = self.timer['iter']:time().real
    
    self:iterPrint(set, epoch, iBatch, loader:nBatch(), dataTime, iterTime)
    self.timer['iter']:reset()                  -- reset timer
    self.timer['data']:reset()
  end
  self:summary(set, epoch)

  if self.option.example ~= nil and
    (self.option.example.type == 'pred' or self.option.example.type == 'allpred'
     or self.option.example.type == 'all') then
    self:example(set, loader)
  end
  self.network:clearState()                   -- clear intermediate module to reduce memory cost
end

function Supervised:test(epoch, loader)
  local set = 'test'
  self:modeFlag(set)                     -- set evaluation flag for BN, dropout
  self:statistics(set, loader) -- reset whatever statistics for test phase in records

  for iBatch, batchData in loader:iterator() do
    self:copyBatch(batchData)              -- inline copy batch to GPU if use
    local dataTime = self.timer['data']:time().real

    self:resetState(epoch, batchData.size) -- save current epoch and batch size, reset used for sequence models
    self:forward('test')
    local iterTime = self.timer['iter']:time().real
    
    self:iterPrint('test', epoch, iBatch, loader:nBatch(), dataTime, iterTime)
    self.timer['iter']:reset()                 -- reset timer
    self.timer['data']:reset()
  end
  self:summary(set, epoch)
  
  if self.option.example ~= nil and
    (self.option.example.type == 'pred' or self.option.example.type == 'allpred'
     or self.option.example.type == 'all') then
    self:example(set, loader)
  end

  self:modeFlag('train')                    -- set default flag to training
  self.network:clearState()                   -- clear intermediate module to reduce memory cost
  return self:measure()
end

