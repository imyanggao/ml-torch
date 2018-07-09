
require('torch')
require('dataset.init')

loader = {}

require('loader.AbstractLoader')
require('loader.SimpleLoader')
require('loader.MTLoader')

function loader.setup(option)
  local trainSet, validSet = dataset.setup(option)
  local trainLoader, validLoader = loader[option.loader.type](trainSet, option)
  if option.lrFinder == false then
    validLoader = loader[option.loader.type](validSet, option)
  end
  option.optim.nTrainBatch = trainLoader:nBatch()
  return trainLoader, validLoader
end

return loader
