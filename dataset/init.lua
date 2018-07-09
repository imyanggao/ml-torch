
-- require('torch')

dataset = {}

require('dataset.Cache')
require('dataset.CacheClsCIFAR')
require('dataset.CacheClsILSVRC')
require('dataset.CacheSegVOC')
require('dataset.CacheSegBRIC')
require('dataset.CacheSegIBSR')
require('dataset.CacheSegIBIS')

require('dataset.Dataset')
require('dataset.DatasetClsCIFAR')
require('dataset.DatasetClsILSVRC')
require('dataset.DatasetSegVOC')
require('dataset.DatasetSegBRIC')
require('dataset.DatasetSegIBSR')
require('dataset.DatasetSegIBIS')
require('dataset.DatasetSegSimple')

function dataset.setup(option)
  local hasTime, has3D, ext, configFile, key = false, false, ''
  if option.data.setConfig ~= nil then
    configFile = option.data.setConfig
  else
    if option.data.set == 'cifar10' then
      key, ext = 'ClsCIFAR', '10'
      option.data.color = true
    elseif option.data.set == 'cifar100' then
      key, ext = 'ClsCIFAR', '100'
      option.data.color = true
    elseif option.data.set == 'imagenet' then
      key = 'ClsILSVRC'
      option.data.color = true
    elseif option.data.set == 'voc' then
      key = 'SegVOC'
      option.data.color = true
    elseif option.data.set == 'voc-sbd' then
      key, ext = 'SegVOC', '-SBD'
      option.data.color = true
    elseif option.data.set == 'bric' then
      key, has3D, hasTime = 'SegBRIC', true, true
      option.data.color = false
    elseif option.data.set == 'ibis' then
      key, has3D, hasTime = 'SegIBIS', true, true
      option.data.color = false
    elseif option.data.set == 'ibsr' then
      key, has3D = 'SegIBSR', true
      option.data.color = false
    else
      error('dataset options: cifar10 | cifar100 | imagenet | voc | voc-sbd | bric | ibis | ibsr')
    end
    configFile = 'dataset/Config' .. key
    
    if has3D == true then
      if option.data.dim == 2 then
        configFile = configFile .. '-2D'
      elseif option.data.dim == 3 then
        configFile = configFile .. '-3D'
      else
        error('dataDim options: 2 | 3 to use as 2d or 3d image')
      end
    end

    if hasTime == true then
      if option.data.time == true then
        configFile = configFile .. 'Time'
      end
    end

    key = 'Dataset' .. key
    configFile = configFile .. ext .. '.txt'
    assert(paths.filep(configFile), configFile .. 'does not exist')
  end
  local config = utility.tbl.load(configFile)
  config.preprocess = option.data.preprocess
  local trainSet, validSet = dataset[key]('train', config)
  if option.lrFinder == false then
    validSet = dataset[key]('val', config)
  end
  option.data.imageSize = trainSet:imageSize()
  option.data.nClass = #trainSet:labels()
  if option.model.weightedLoss == true then
    option.data.hist = trainSet:histLabels()
  end
  
  if string.find(option.data.set, 'voc') then
    option.data.ignoreIndex = 22
    option.data.colormap = trainSet.Colormap
  end
  if option.data.ignoreIndex > 0 then
    local indices = torch.range(1, option.data.nClass):totable()
    indices[option.data.ignoreIndex] = nil
    option.data.nClass = option.data.nClass - 1
    if option.model.weightedLoss == true then
      option.data.hist = option.data.hist:index(1, torch.LongTensor(indices))
    end
  end
  
  return trainSet, validSet
end

return dataset
