
local AbstractLoader = torch.class('loader.AbstractLoader')

function AbstractLoader:__init()
end

function AbstractLoader:nBatch()
  return math.ceil(self.sampleSz / self.batchSz)
end

function AbstractLoader:sampleSize()
  return self.sampleSz
end

-- function AbstractLoader:histLabels()
--   return self.hist
-- end

function AbstractLoader:classes()
  return self.clses
end

function AbstractLoader:labels()
  return self.lbls
end

function AbstractLoader:imageSize()
  return self.imageSz
end

function AbstractLoader:iterator()
end

