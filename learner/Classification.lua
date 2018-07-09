
local Classification, parent = torch.class('learner.Classification', 'learner.Supervised')

function Classification:__init(model, criterion, option)
  parent.__init(self, model, criterion, option)
end

function Classification:confusionUpdate(set)
  self.confusion[set]:updateValids()      -- update valids, i.e. acc, iou
  if set == 'train' then
    return ' | Epoch: '
  elseif set == 'test' then
    return ' | Test: '
  end
end

function Classification:iterPrint(set, epoch, iBatch, nBatch, dataTime, iterTime)
  local headStr = self:confusionUpdate(set)
  local acc, iou = self.confusion[set].averageValid, self.confusion[set].averageUnionValid
  print((headStr .. '[%d][%d/%d]    Time %.2f  Data %.2f    Grad %6.4f    Loss %6.4f    ACC %6.4f    IOU %6.4f')
      :format(epoch, iBatch, nBatch, iterTime, dataTime,
              self.gradParams:norm() / self.params:norm(), self.loss['iter'], acc, iou))
  
  -- if set == 'train' then
  --   if epoch == self.option.optim.iEpoch and iBatch == 1 then
  --     local ext = '.txt'
  --     self.paramsLog = {}
  --     for _, node in ipairs(self.model.forwardnodes) do
  --       local name = node.data.annotations.name
  --       local model = node.data.module
  --       if name ~= nil and string.find(name, 'conv1_1') then
  --         print(name)
  --         self.paramsLog[name] = optim.Logger(self.option.log .. '-' .. name .. ext)
  --         local weightSize = model:parameters()[1]:size()
  --         local biasSize = model:parameters()[2]:size()
  --         local legends = {'epoch', 'iter'}
  --         for i = 1, weightSize[1] do
  --           for j = 1, weightSize[2] do
  --             for k = 1, weightSize[3] do
  --               for l = 1, weightSize[4] do
  --                 table.insert(legends, 'w' .. i .. '-' .. j .. '-' .. k .. '-' .. l)
  --               end
  --             end
  --           end
  --         end
  --         for i = 1, biasSize[1] do
  --           table.insert(legends, 'b' .. i)
  --         end
  --         for i = 1, weightSize[1] do
  --           for j = 1, weightSize[2] do
  --             for k = 1, weightSize[3] do
  --               for l = 1, weightSize[4] do
  --                 table.insert(legends, 'dw' .. i .. '-' .. j .. '-' .. k .. '-' .. l)
  --               end
  --             end
  --           end
  --         end
  --         for i = 1, biasSize[1] do
  --           table.insert(legends, 'db' .. i)
  --         end
  --         self.paramsLog[name]:setNames(legends)
  --       end
  --     end
  --   end

  --   for _, node in ipairs(self.model.forwardnodes) do
  --     local name = node.data.annotations.name
  --     local model = node.data.module
  --     if name ~= nil and string.find(name, 'conv1_1') then
  --       local m = torch.Tensor({epoch, iBatch}):typeAs(self.model)
  --       local p, dp = model:parameters()
  --       m = torch.cat(m, p[1]:view(-1))
  --       m = torch.cat(m, p[2]:view(-1))
  --       m = torch.cat(m, dp[1]:view(-1))
  --       m = torch.cat(m, dp[2]:view(-1))
  --       self.paramsLog[name]:add(m:totable())
  --     end
  --   end
  -- end
  
end

function Classification:statistics(set, loader)
  if self.confusion ==  nil then
    local savedLabels = loader:labels()
    if self.option.data.ignoreIndex > 0 then -- if exists ignored class
      savedLabels[self.option.data.ignoreIndex] = nil
    end
    self.confusion = {['train'] = optim.ConfusionMatrix(savedLabels),
      ['test'] = optim.ConfusionMatrix(savedLabels)}
  end
  self.confusion[set]:zero()    -- reset confusion matrix
  parent.statistics(self, set, loader)
end

function Classification:statisticsUpdate(set)
  parent.statisticsUpdate(self, set)
  local output, target = self.model.output, self.target
  if self.option.data.ignoreIndex > 0 then -- if exists ignored class
    local validIndices = target:ne(self.option.data.ignoreIndex)
    output = output[validIndices:view(validIndices:nElement(), 1):expandAs(output)]
    if validIndices:sum() ~= 0 then -- in case of all labels are equal to ignoreIndex
      output = output:view(validIndices:sum(), -1)
    end
    target = target[validIndices]
  end
  if target:nElement() ~= 0 then
    self.confusion[set]:batchAdd(output, target)
  end
end

-- function Classification:summary(epoch)
--   parent.summary(self, epoch)
--   self:confusionUpdate('train')
--   self:confusionUpdate('test')
--   print(self.confusion['train'].mat)
--   print(self.confusion['test'].mat)
--   local trainAcc, trainIOU = self.confusion['train'].averageValid, self.confusion['train'].averageUnionValid
--   local testAcc, testIOU = self.confusion['test'].averageValid, self.confusion['test'].averageUnionValid
--   print((sys.COLORS.green .. ' * Finished epoch # %d     ' .. sys.COLORS.red .. 'ACC %6.4f    IOU %6.4f')
--       :format(epoch, testAcc, testIOU))
--   local labels = self.confusion['test'].classes
--   if self.mainLogger == nil then
--     local ext = '.log'
--     local mainLogPath = self.option.log .. '-log' .. ext
--     local accLogPath = self.option.log .. '-acc' .. ext
--     local iouLogPath = self.option.log .. '-iou' .. ext
--     self.mainLogger = optim.Logger(mainLogPath)
--     self.accLogger = optim.Logger(accLogPath)
--     self.iouLogger = optim.Logger(iouLogPath)
--     self.mainLogger:setNames({'epoch  ', 'lr      ', '[loss] ', '<loss>  ', '[acc] ', '<acc>   ', '[iou] ', '<iou>   '})
--     local names = {'epoch  '}
--     for i = 1, #labels do
--       local label = utility.str.fixLen(labels[i], 4)
--       table.insert(names, '[' .. label .. ']')
--       table.insert(names, '<' .. label .. '>')
--     end
--     self.accLogger:setNames(names)
--     self.iouLogger:setNames(names)
--   end
--   self.mainLogger:add({string.format('%-7d', epoch),
--                        string.format('%-8.6f', self.option.optim.state.learningRate),
--                        string.format('%-7.4f', self.loss['train']), 
--                        string.format('%-8.4f', self.loss['test']),
--                        string.format('%-6.4f', trainAcc),
--                        string.format('%-8.4f', testAcc),
--                        string.format('%-6.4f', trainIOU),
--                        string.format('%-8.4f', testIOU)})
--   local accLog, iouLog = {string.format('%-7d', epoch)}, {string.format('%-7d', epoch)}
--   for i = 1, #labels do
--     table.insert(accLog, string.format('%-7.4f', self.confusion['train'].valids[i]))
--     table.insert(accLog, string.format('%-7.4f', self.confusion['test'].valids[i]))
--     table.insert(iouLog, string.format('%-7.4f', self.confusion['train'].unionvalids[i]))
--     table.insert(iouLog, string.format('%-7.4f', self.confusion['test'].unionvalids[i]))
--   end
--   self.accLogger:add(accLog)
--   self.iouLogger:add(iouLog)
--   if self.pastalog ~= nil then
--     local _, base, _ = utility.io.splitPath(self.option.log)
--     self.pastalog(base, 'trainAcc', trainAcc, epoch, self.option.url)
--     self.pastalog(base, 'validAcc', testAcc, epoch, self.option.url)
--     self.pastalog(base, 'trainIOU', trainIOU, epoch, self.option.url)
--     self.pastalog(base, 'validIOU', testIOU, epoch, self.option.url)
--   end
-- end

function Classification:summary(set, epoch)
  parent.summary(self, set, epoch)
  self:confusionUpdate(set)
  print(self.confusion[set])
  print((sys.COLORS.green .. ' * ' .. set .. ' epoch # %d     ' .. sys.COLORS.red .. 'ACC %6.4f    IOU %6.4f')
      :format(epoch, self.confusion[set].averageValid, self.confusion[set].averageUnionValid))
  if set == 'test' then
    if self.mainLogger == nil then
      self.mainLogger = optim.Logger(self.option.log .. '-log', true)
      self.accLogger = optim.Logger(self.option.log .. '-acc', true)
      self.iouLogger = optim.Logger(self.option.log .. '-iou', true)
      self.mainLogger:setNames({'epoch  ', 'lr      ', '[loss] ', '<loss>  ', '[acc] ', '<acc>   ', '[iou] ', '<iou>   '})
      local names = {'epoch  '}
      for i = 1, #self.confusion[set].classes do
        local label = utility.str.fixLen(self.confusion[set].classes[i], 4)
        table.insert(names, '[' .. label .. ']')
        table.insert(names, '<' .. label .. '>')
      end
      self.accLogger:setNames(names)
      self.iouLogger:setNames(names)
    end
    self.mainLogger:add({string.format('%-7d', epoch),
                         string.format('%-8.6f', self.option.optim.state[1].learningRate),
                         string.format('%-7.4f', self.loss['train']), 
                         string.format('%-8.4f', self.loss['test']),
                         string.format('%-6.4f', self.confusion['train'].averageValid),
                         string.format('%-8.4f', self.confusion['test'].averageValid),
                         string.format('%-6.4f', self.confusion['train'].averageUnionValid),
                         string.format('%-8.4f', self.confusion['test'].averageUnionValid)})
    local accLog, iouLog = {string.format('%-7d', epoch)}, {string.format('%-7d', epoch)}
    for i = 1, #self.confusion[set].classes do
      table.insert(accLog, string.format('%-7.4f', self.confusion['train'].valids[i]))
      table.insert(accLog, string.format('%-7.4f', self.confusion['test'].valids[i]))
      table.insert(iouLog, string.format('%-7.4f', self.confusion['train'].unionvalids[i]))
      table.insert(iouLog, string.format('%-7.4f', self.confusion['test'].unionvalids[i]))
    end
    self.accLogger:add(accLog)
    self.iouLogger:add(iouLog)
  end
  
  if self.pastalog ~= nil then
    local _, base, _ = utility.io.splitPath(self.option.log)
    self.pastalog(base, set .. 'Acc', self.confusion[set].averageValid, epoch, self.option.url)
    self.pastalog(base, set .. 'IOU', self.confusion[set].averageUnionValid, epoch, self.option.url)
  end
end

function Classification:example(set, loader)
  if self.option.example.set ~= nil then
    parent.example(self, set, loader)
  end
end

function Classification:measure()
  return self.confusion['test'].averageUnionValid
end
