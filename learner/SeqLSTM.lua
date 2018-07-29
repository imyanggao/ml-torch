
local SeqLSTM, parent = torch.class('learner.SeqLSTM', 'learner.Sequence')

function SeqLSTM:__init(model, criterion, ntime, CHSz, option)
  parent.__init(self, model, criterion, ntime, option)
  self.CHSz = CHSz
  self:prepareModel()
end

function SeqLSTM:prepareModel()
  self.nCH = #self.CHSz * 2
  self.CH = {}
  self.gradCH = {}
  for t = 0, self.ntime do
    self.CH[t] = {}
    self.gradCH[t] = {}
  end
  self.networks = utility.net.sharedClone(self.network, self.ntime)
  self.criterions = utility.net.sharedClone(self.criterion, self.ntime)
  collectgarbage()
end

-- function SeqLSTM:copyBatch(batchData)
--   self.input:resize(batchData.input:size()):copy(batchData.input)
--   self.target:resize(batchData.target:size()):copy(batchData.target)
--   assert(self.ntime == self.input:size(1), 'ntime in current batch is different with initialized ntime')
-- end

-- !!!big bug exits in old code function create_init_hc, because all zero are shared
-- Always reset zero when starting a new epoch
-- Between iterations within an epoch, using self.resetStateZero to decide either reset zero or clone latest
function SeqLSTM:resetState(curEpoch, curBatchSz)
  local function CHCreateZero()
    for i = 1, self.nCH/2 do
      local zero = torch.zeros(curBatchSz, table.unpack(self.CHSz[i])):typeAs(self.network) -- for GPU
      self.CH[0][(i-1)*2+1] = zero:clone() -- for cell state
      self.CH[0][(i-1)*2+2] = zero:clone() -- for hidden state
    end
  end
  local function CHFillZero()
    for i = 1, self.nCH/2 do
      self.CH[0][(i-1)*2+1]:zero()
      self.CH[0][(i-1)*2+2]:zero()
    end
  end
  local function CHClone()
    for i = 1, self.nCH/2 do
      self.CH[0][(i-1)*2+1] = self.CH[0][(i-1)*2+1][{{1,curBatchSz}}]
      self.CH[0][(i-1)*2+2] = self.CH[0][(i-1)*2+2][{{1,curBatchSz}}]
      self.CH[0][(i-1)*2+1]:clone(self.CH[self.ntime][(i-1)*2+1][{{1,curBatchSz}}])
      self.CH[0][(i-1)*2+2]:clone(self.CH[self.ntime][(i-1)*2+2][{{1,curBatchSz}}])
    end
  end
  
  if self.latest['epoch'] == 0 then           -- before 1st iteration
    CHCreateZero()
  else
    if curBatchSz > self.latest['batchSz'] then -- only happens if beginning a new epoch, reset zero
      CHCreateZero()
    elseif curBatchSz == self.latest['batchSz'] then
      if curEpoch ~= self.latest['epoch'] then  -- if beginning of epoch, reset zero
        CHFillZero()
      else                                  -- if between iterations in a epoch, 
        if self.resetStateZero ~= nil then  -- either reset zero,
          CHFillZero()
        else                                -- or clone latest state, depends on self.resetStateZero argument
          CHClone()
        end
      end
    else                                    -- only happens at the last iteration in an epoch or test with small batch
      if self.resetStateZero == nil then    -- if clone latest state with cut
        CHClone()
      else
        CHCreateZero()                      -- reset zero
      end
    end
  end

  self.gradCH[self.ntime] = utility.tbl.tsrClone(self.CH[0], true) -- always reset gradient at last time to be zero

  parent.resetState(self, curEpoch, curBatchSz)
end

function SeqLSTM:forward(set)
  self.loss['iter'][0] = 0
  for t = 1, self.ntime do
    print(self.CH[t-1])
    print(self.input[t]:size())
    self.networks[t]:forward(utility.tbl.cat(self.CH[t-1], self.input[t]))
    for i = 1, self.nCH do
      self.CH[t][i] = self.networks[t].output[i]
    end
    self.loss['iter'][t] = self.criterions[t]:forward(self.networks[t].output[self.nCH+1], self.target[t])
    self:statisticsUpdate(set, t)
  end
  self.loss['iter'][0] = self.loss['iter'][0] / self.ntime
end

function SeqLSTM:backward()
  for t = self.ntime, 1, -1 do
    self.criterions[t]:backward(self.networks[t].output[self.nCH+1], self.target[t])
    self.networks[t]:backward(utility.tbl.cat(self.CH[t-1], self.input[t]),
                            utility.tbl.cat(self.gradCH[t], self.criterions[t].gradInput))
    for i = 1, self.nCH do
      self.gradCH[t-1][i] = self.networks[t].gradInput[i]
    end
    -- clear intermediate module states as output, gradInput, etc. for low gpu memory.
    if t ~= self.ntime then
      self.networks[t]:clearState()
    end
  end
end
