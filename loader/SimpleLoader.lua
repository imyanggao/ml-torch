
local SimpleLoader, parent = torch.class('loader.SimpleLoader', 'loader.AbstractLoader')

function SimpleLoader:__init(dataset, option)
  self.dataset = dataset
  self.option = option
  self.clses = dataset:classes()
  self.lbls = dataset:labels()
  self.sampleSz = dataset:size()
  -- self.imageSz = dataset:imageSize()
  -- self.hist = dataset:histLabels()
  self.batchSz = option.loader.batchSize
end

function SimpleLoader:iterator()
  -- warning: when sampleSz > 10^7, torch.randperm causes out of memory for luajit
  local perm = torch.randperm(self.sampleSz)
  local idx, iBatch, batchData = 1, 0, nil
  local function nextBatch()
    iBatch = iBatch + 1
    while idx <= self.sampleSz do
      local indices = perm:narrow(1, idx, math.min(self.batchSz, self.sampleSz - idx + 1))
      local curBatchSz = indices:numel()
      local batchData, batchLabel, inputSz, targetSz
      for i, index in ipairs(indices:totable()) do
        local sample = self.dataset:get(index)
        local input, target = self.dataset:preprocess(sample, self.option)
        if i == 1 then
          inputSz = input:size():totable()
          -- 2D image: channel * height * width
          -- 3D image: channel * slice * height * width
          -- 2D image + time: time * channel * height * width
          -- 3D image + time: time * channel * slice * height * width
          batchData = torch[self.option.data.inputType](curBatchSz, table.unpack(inputSz))
          if type(target) == 'number' then -- classification with integer label
            targetSz = {nil}
          else
            targetSz = target:size():totable()
          end
          batchLabel = torch[self.option.data.targetType](curBatchSz, table.unpack(targetSz))
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
      idx = idx + curBatchSz
      collectgarbage()
      return iBatch, {input = batchData, target = batchLabel, size = curBatchSz}
    end
    return nil
  end
  return nextBatch
end
