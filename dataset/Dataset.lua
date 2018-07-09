
local Dataset = torch.class('dataset.Dataset')

function Dataset:__init(split, opt)
end

function Dataset:get(i)
end

function Dataset:size()
end

function Dataset:classes()
  return self.clses
end

function Dataset:labels()
  return self.lbls
end

function Dataset:time()
  return self.tm
end

function Dataset:histLabels()
end

-- change the size according to the output of preprocess
function Dataset:imageSize()
end

function Dataset:preprocess(sample)
  return sample.input, sample.target
end
