
local Sequence, parent = torch.class('learner.Sequence', 'learner.Supervised')

function Sequence:__init(model, criterion, ntime, option)
  parent.__init(self, model, criterion, option)
  self.ntime = ntime
end

-- function Sequence:prepareModel()
--   print('parent')
-- end

function Sequence:modeFlag(set)
  if set == 'train' then
    for t = 1, self.ntime do
      self.networks[t]:training()
    end
  elseif set == 'test' then
    for t = 1, self.ntime do
      self.networks[t]:evaluate()
    end
  end
end

function Sequence:copyBatch(batchData)
  self.input:resize(batchData.input:size()):copy(batchData.input)
  self.target:resize(batchData.target:size()):copy(batchData.target)
  print(self.input:size())
  print(self.target:size())
  assert(self.ntime == self.input:size(1), 'ntime in current batch is different with initialized ntime')
end

function Sequence:statistics(set, loader)
  if self.latest == nil then
    self.latest = {['epoch'] = 0, ['batchSz'] = 0}
  end
  if self.sampleSz == nil then
    self.sampleSz = {}
  end
  if self.sampleSz[set] == nil then
    self.sampleSz[set] = loader:sampleSize()
  end
  if self.timer == nil then
    self.timer = {['data'] = torch.Timer(), ['iter'] = torch.Timer()} -- timer for each iteration and loading data
  end
  if self.loss == nil then
    self.loss = {['train'] = {}, ['test'] = {}, ['iter'] = {}}
  end
  self.timer['iter']:reset()    -- reset iteration timer
  self.timer['data']:reset()    -- reset data processing timer
  for t = 0, self.ntime do
    self.loss[set][t] = 0            -- reset loss
  end
end

function Sequence:statisticsUpdate(set, t)
  self.loss['iter'][0] = self.loss['iter'][0] + self.loss['iter'][t] * self.latest['batchSz']
  self.loss[set][0] = self.loss[set][0] + self.loss['iter'][t] * self.latest['batchSz']
  self.loss[set][t] = self.loss[set][t] + self.loss['iter'][t] * self.latest['batchSz']
end

-- function Sequence:summary(epoch)
--   for i,set in pairs({'train', 'test'}) do
--     self.loss[set][0] = self.loss[set][0] / (self.sampleSz[set] * self.ntime)
--     for t = 1, self.ntime do
--       self.loss[set][t] = self.loss[set][t] / self.sampleSz[set]
--     end
--   end
--   if self.pastalog ~= nil then
--     local _, base, _ = utility.io.splitPath(self.option.log)
--     for t = 0, self.ntime do
--       self.pastalog(base, 'trainLoss[' .. t .. ']', self.loss['train'][t], epoch, self.option.url)
--       self.pastalog(base, 'validLoss[' .. t .. ']', self.loss['test'][t], epoch, self.option.url)
--     end
--   end
-- end

function Sequence:summary(set, epoch)
  self.loss[set][0] = self.loss[set][0] / (self.sampleSz[set] * self.ntime)
  for t = 1, self.ntime do
    self.loss[set][t] = self.loss[set][t] / self.sampleSz[set]
  end
  if self.pastalog ~= nil then
    local _, base, _ = utility.io.splitPath(self.option.log)
    for t = 0, self.ntime do
      self.pastalog(base, set .. 'Loss[' .. t .. ']', self.loss[set][t], epoch, self.option.url)
    end
  end
end

function Sequence:measure()
  return self.loss['test'][0]
end
