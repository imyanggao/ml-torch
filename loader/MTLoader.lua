
local MTLoader, parent = torch.class('loader.MTLoader', 'loader.AbstractLoader')

function MTLoader:__init(dataset, option)
  self.dataset = dataset
  self.option = option
  self.clses = dataset:classes()
  self.lbls = dataset:labels()
  self.sampleSz = dataset:size()
  -- self.imageSz = dataset:imageSize()
  -- self.hist = dataset:histLabels()
  self.batchSz = option.loader.batchSize
  -- claim the threads pool
  self.threads = require('threads')
  self.threads.Threads.serialization('threads.sharedserialize')
  self.pool = self.threads.Threads(
    self.option.loader.nThreads,
    -- init function:
    -- to make upvalue dataset as known type
    -- before deserializate next main function for each thread
    function()
      require('dataset.init')
    end,
    -- main function:
    -- only local variable dataset could be upvalue, not self.dataset
    -- a global variable __dataset is need for each thread
    function(iThread)
      print('start dataloader thread[' .. iThread .. ']')
      __dataset = dataset
      __option = option
    end
  )
end

function MTLoader:examples(indices)
  local nExamples = utility.tbl.len(indices)
  for i, index in ipairs(indices) do
    if index > 0 and index <= self.sampleSz then
      local sample = self.dataset:get(index)
      local input, target = self.dataset:preprocess(sample, self.option, 'val')
      if i == 1 then
        inputSz = input:size():totable()
        exampleData = torch.Tensor(nExamples, table.unpack(inputSz)):typeAs(input)
        if type(target) == 'number' then
          targetSz = {nil}
          exampleLabel = torch.Tensor(nExamples, table.unpack(targetSz))
        else
          targetSz = target:size():totable()
          exampleLabel = torch.Tensor(nExamples, table.unpack(targetSz)):typeAs(target)
        end
      end
      exampleData[i]:copy(input)
      if type(target) == 'number' then
        exampleLabel[i] = target
      else
        exampleLabel[i]:copy(target)
      end
    else
      error(sys.COLORS.red .. 'exampleIndices should be between [1, sampleSz]')
    end
  end
  if self.dataset:time() ~= nil then
    exampleData = exampleData:transpose(1, 2):contiguous()
    exampleLabel = exampleLabel:transpose(1, 2):contiguous()
  end
  return {input = exampleData, target = exampleLabel}
end

function MTLoader:iterator()
  -- warning: when sampleSz > 10^7, torch.randperm causes out of memory for luajit
  local perm = torch.randperm(self.sampleSz)
  local idx, iBatch, batchData = 1, 0, nil
  local function queue()
    while idx <= self.sampleSz and self.pool:acceptsjob() do
      local indices = perm:narrow(1, idx, math.min(self.batchSz, self.sampleSz - idx + 1))
      self.pool:addjob(
        function(indices)
          local curBatchSz = indices:numel()
          local batchData, batchLabel, inputSz, targetSz
          for i, index in ipairs(indices:totable()) do
            local sample = __dataset:get(index)
            local input, target = __dataset:preprocess(sample, __option)
            if i == 1 then
              inputSz = input:size():totable()
              -- 2D image: channel * height * width
              -- 3D image: channel * slice * height * width
              -- 2D image + time: time * channel * height * width
              -- 3D image + time: time * channel * slice * height * width
              batchData = torch.Tensor(curBatchSz, table.unpack(inputSz)):typeAs(input)
              if type(target) == 'number' then -- classification with integer label
                targetSz = {nil}
                batchLabel = torch.Tensor(curBatchSz, table.unpack(targetSz))
              else
                targetSz = target:size():totable()
                batchLabel = torch.Tensor(curBatchSz, table.unpack(targetSz)):typeAs(target)
              end              
            end
            batchData[i]:copy(input)
            if type(target) == 'number' then
              batchLabel[i] = target
            else
              batchLabel[i]:copy(target)
            end
          end
          if __dataset:time() ~= nil then -- for dataset with time, batch format: time * nBatch * channel * ...
            batchData = batchData:transpose(1, 2):contiguous()
            batchLabel = batchLabel:transpose(1, 2):contiguous()
          end
          collectgarbage()
          return {input = batchData, target = batchLabel, size = curBatchSz, index = indices}
        end,
        function(sampleBatch)
          batchData = sampleBatch
        end,
        indices
      )
      idx = idx + self.batchSz
    end
    iBatch = iBatch + 1
    if self.pool:hasjob() then
      self.pool:dojob()
      return iBatch, batchData
    else
      return nil
    end
  end
  return queue
end
