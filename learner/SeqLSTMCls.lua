
local SeqLSTMCls, parent = torch.class('learner.SeqLSTMCls', 'learner.SeqLSTM')

function SeqLSTMCls:__init(model, criterion, ntime, CHSz, option)
  parent.__init(self, model, criterion, ntime, CHSz, option)
end

function SeqLSTMCls:copyBatch(batchData)
  -- batchData.input = batchData.input:transpose(3,4):transpose(4,5):contiguous():view(self.ntime, -1, 2)
  -- batchData.target = batchData.target:view(self.ntime, -1)
  -- batchData.size = batchData.target:size(2)
  self.input:resize(batchData.input:size()):copy(batchData.input)
  self.target:resize(batchData.target:size()):copy(batchData.target)
  self.target = self.target:view(self.ntime, -1)
  assert(self.ntime == self.input:size(1), 'ntime in current batch is different with initialized ntime')
end

function SeqLSTMCls:confusionUpdate(set)
  for t = 0, self.ntime do
    self.confusion[set][t]:updateValids()      -- update valids, i.e. acc, iou
  end
  if set == 'train' then
    return ' | Epoch: '
  elseif set == 'test' then
    return ' | Test: '
  end
end

function SeqLSTMCls:iterPrint(set, epoch, iBatch, nBatch, dataTime, iterTime)
  local headStr = self:confusionUpdate(set)
  local acc, iou = self.confusion[set][0].averageValid, self.confusion[set][0].averageUnionValid
  print((headStr .. '[%d][%d/%d]    Time %.2f  Data %.2f    Grad %6.4f    Loss %1.4f    ACC %6.4f    IOU %6.4f')
      :format(epoch, iBatch, nBatch, iterTime, dataTime,
              self.gradParams:norm() / self.params:norm(), self.loss['iter'][0], acc, iou))
end

function SeqLSTMCls:statistics(set, loader)
  if self.confusion == nil then
    self.confusion = {['train'] = {}, ['test'] = {}}
    for t = 0, self.ntime do
      self.confusion['train'][t] = optim.ConfusionMatrix(loader:labels())
      self.confusion['test'][t] = optim.ConfusionMatrix(loader:labels())
    end
  end
  for t = 0, self.ntime do
    self.confusion[set][t]:zero()
  end
  parent.statistics(self, set, loader)
end

function SeqLSTMCls:statisticsUpdate(set, t)
  parent.statisticsUpdate(self, set, t)
  self.confusion[set][0]:batchAdd(self.models[t].output[self.nCH+1], self.target[t])
  self.confusion[set][t]:batchAdd(self.models[t].output[self.nCH+1], self.target[t])
end

-- function SeqLSTMCls:summary(epoch)
--   parent.summary(self, epoch)
--   self:confusionUpdate('train')
--   self:confusionUpdate('test')
--   for t = 0, self.ntime do
--     local trainAcc, trainIOU = self.confusion['train'][t].averageValid, self.confusion['train'][t].averageUnionValid
--     local testAcc, testIOU = self.confusion['test'][t].averageValid, self.confusion['test'][t].averageUnionValid
--     if t == 0 then
--       print((sys.COLORS.green .. ' * Finished epoch # %d     ' .. sys.COLORS.red .. 'ACC %6.4f    IOU %6.4f')
--           :format(epoch, testAcc, testIOU))
--     else
--       print((sys.COLORS.green .. '             at time %d    ' .. sys.COLORS.red .. 'ACC %6.4f    IOU %6.4f')
--           :format(t, testAcc, testIOU))
--     end
--     local labels, timeStr = self.confusion['test'][t].classes, ''
--     if t ~= 0 then
--       timeStr = '-t' .. t
--     end
--     if self.mainLogger == nil then
--       local ext = '.log'
--       local mainLogPath = self.option.log .. '-log' .. timeStr .. ext
--       local accLogPath = self.option.log .. '-acc' .. timeStr .. ext
--       local iouLogPath = self.option.log .. '-iou' .. timeStr .. ext
--       if t == 0 then
--         self.mainLogger, self.accLogger, self.iouLogger = {}, {}, {}
--       end
--       self.mainLogger[t] = optim.Logger(mainLogPath)
--       self.accLogger[t] = optim.Logger(accLogPath)
--       self.iouLogger[t] = optim.Logger(iouLogPath)
--       self.mainLogger[t]:setNames({'epoch  ', 'lr      ', '[loss] ', '<loss>  ', '[acc] ', '<acc>   ', '[iou] ', '<iou>   '})
--       local names = {'epoch  '}
--       for i = 1, #labels do
--         local label = utility.str.fixLen(labels[i], 4)
--         table.insert(names, '[' .. label .. ']')
--         table.insert(names, '<' .. label .. '>')
--       end
--       self.accLogger[t]:setNames(names)
--       self.iouLogger[t]:setNames(names)
--     end
--     self.mainLogger[t]:add({string.format('%-7d', epoch),
--                             string.format('%-8.6f', self.option.optim.state.learningRate),
--                             string.format('%-7.4f', self.loss['train'][t]), 
--                             string.format('%-8.4f', self.loss['test'][t]),
--                             string.format('%-6.4f', trainAcc),
--                             string.format('%-8.4f', testAcc),
--                             string.format('%-6.4f', trainIOU),
--                             string.format('%-8.4f', testIOU)})
--     local accLog, iouLog = {string.format('%-7d', epoch)}, {string.format('%-7d', epoch)}
--     for i = 1, #labels do
--       table.insert(accLog, string.format('%-7.4f', self.confusion['train'][t].valids[i]))
--       table.insert(accLog, string.format('%-7.4f', self.confusion['test'][t].valids[i]))
--       table.insert(iouLog, string.format('%-7.4f', self.confusion['train'][t].unionvalids[i]))
--       table.insert(iouLog, string.format('%-7.4f', self.confusion['test'][t].unionvalids[i]))
--     end
--     self.accLogger[t]:add(accLog)
--     self.iouLogger[t]:add(iouLog)
--     if self.pastalog ~= nil then
--       local _, base, _ = utility.io.splitPath(self.option.log)
--       self.pastalog(base, 'trainAcc[' .. t .. ']', trainAcc, epoch, self.option.url)
--       self.pastalog(base, 'validAcc[' .. t .. ']', testAcc, epoch, self.option.url)
--       self.pastalog(base, 'trainIOU[' .. t .. ']', trainIOU, epoch, self.option.url)
--       self.pastalog(base, 'validIOU[' .. t .. ']', testIOU, epoch, self.option.url)
--     end
--   end
-- end

function SeqLSTMCls:summary(set, epoch)
  parent.summary(self, set, epoch)
  self:confusionUpdate(set)
  for t = 0, self.ntime do
    if t == 0 then
      print((sys.COLORS.green .. ' * ' .. set .. ' epoch # %d     ' .. sys.COLORS.red .. 'ACC %6.4f    IOU %6.4f')
          :format(epoch, self.confusion[set][t].averageValid, self.confusion[set][t].averageUnionValid))
    else
      print((sys.COLORS.green .. '             at time %d    ' .. sys.COLORS.red .. 'ACC %6.4f    IOU %6.4f')
          :format(t, self.confusion[set][t].averageValid, self.confusion[set][t].averageUnionValid))
    end
    if set == 'test' then
      local timeStr = ''
      if t ~= 0 then
        timeStr = '-t' .. t
      end
      if self.mainLogger == nil then
        if t == 0 then
          self.mainLogger, self.accLogger, self.iouLogger = {}, {}, {}
        end
        self.mainLogger[t] = optim.Logger(self.option.log .. '-log' .. timeStr, true)
        self.accLogger[t] = optim.Logger(self.option.log .. '-acc' .. timeStr, true)
        self.iouLogger[t] = optim.Logger(self.option.log .. '-iou' .. timeStr, true)
        self.mainLogger[t]:setNames({'epoch  ', 'lr      ', '[loss] ', '<loss>  ', '[acc] ', '<acc>   ', '[iou] ', '<iou>   '})
        local names = {'epoch  '}
        for i = 1, #self.confusion[set][t].classes do
          local label = utility.str.fixLen(self.confusion[set][t].classes[i], 4)
          table.insert(names, '[' .. label .. ']')
          table.insert(names, '<' .. label .. '>')
        end
        self.accLogger[t]:setNames(names)
        self.iouLogger[t]:setNames(names)
      end
      self.mainLogger[t]:add({string.format('%-7d', epoch),
                              string.format('%-8.6f', self.option.optim.state[1].learningRate),
                              string.format('%-7.4f', self.loss['train'][t]), 
                              string.format('%-8.4f', self.loss['test'][t]),
                              string.format('%-6.4f', self.confusion['train'][t].averageValid),
                              string.format('%-8.4f', self.confusion['test'][t].averageValid),
                              string.format('%-6.4f', self.confusion['train'][t].averageUnionValid),
                              string.format('%-8.4f', self.confusion['test'][t].averageUnionValid)})
      local accLog, iouLog = {string.format('%-7d', epoch)}, {string.format('%-7d', epoch)}
      for i = 1, #self.confusion[set][t].classes do
        table.insert(accLog, string.format('%-7.4f', self.confusion['train'][t].valids[i]))
        table.insert(accLog, string.format('%-7.4f', self.confusion['test'][t].valids[i]))
        table.insert(iouLog, string.format('%-7.4f', self.confusion['train'][t].unionvalids[i]))
        table.insert(iouLog, string.format('%-7.4f', self.confusion['test'][t].unionvalids[i]))
      end
      self.accLogger[t]:add(accLog)
      self.iouLogger[t]:add(iouLog)
    end
    
    if self.pastalog ~= nil then
      local _, base, _ = utility.io.splitPath(self.option.log)
      self.pastalog(base, set .. 'Acc[' .. t .. ']', self.confusion[set][t].averageValid, epoch, self.option.url)
      self.pastalog(base, set .. 'IOU[' .. t .. ']', self.confusion[set][t].averageUnionValid, epoch, self.option.url)
    end
  end
end

function SeqLSTMCls:measure()
  return self.confusion['test'][0].averageUnionValid
end
