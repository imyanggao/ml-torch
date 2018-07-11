
local ffi = require 'ffi'
local image = require 'image'
require 'utility.init'

local DatasetSegVOC, parent = torch.class('dataset.DatasetSegVOC', 'dataset.Dataset')

function DatasetSegVOC:__init(split, config)
  assert(split == 'train' or split == 'val', 'split must be train or val') 
  if io.open(config.path['cacheFile'], 'r') == nil then
    local cacheMaker = dataset.CacheSegVOC(config)
    cacheMaker:save()
  end
  local cache = torch.load(config.path['cacheFile'])
  self.set = cache[split]
  self.clses = cache.classes
  self.lbls = cache.labels
  self.split = split
  self.dataDir = config.path['dataDir']
  self.labelDir = config.path['labelDir']
  self.config = config
  self.Colormap = utility.Colormap('voc')
  self.nChannel = 3
end

function DatasetSegVOC:loadImage(path, type)
  local ok, img = pcall(function() return image.load(path, self.nChannel, self.config.tensorType[type]) end)
  if not ok then
    error('error reading: ' .. path)
  end
  return img
end

function DatasetSegVOC:get(i)
  local path = ffi.string(self.set.datapath[i]:data())
  local data = self:loadImage(paths.concat(self.dataDir, path) .. '.jpg', 'data')
  path = ffi.string(self.set.labelpath[i]:data())
  local label = self:loadImage(paths.concat(self.labelDir, path) .. '.png', 'label')
  label = self.Colormap:c2l(label)
  return {input = data:type('torch.' .. self.config.inputType),
          target = label:type('torch.' .. self.config.targetType)}
end

function DatasetSegVOC:size()
  return self.set.datapath:size(1)
end

function DatasetSegVOC:histLabels()
  if self.hist == nil then
    print('Calculate ' .. self.split .. ' labels, this may take some time')
    local labels
    for i = 1, self:size() do
      xlua.progress(i, self:size())
      local sample = self:get(i)
      if i == 1 then
        self.hist = sample.target:double():histc(#self.lbls)
      else
        self.hist = self.hist + sample.target:double():histc(#self.lbls)
      end
    end
  end
  return self.hist
end

function DatasetSegVOC:imageSize()
  return {self.nChannel, 224, 224} -- change the size according to the output of preprocess
end

function DatasetSegVOC:preprocess(sample, option, split)
  split = split or self.split
  local input, target, transforms = sample.input, sample.target
  -- target[target:eq(22)] = 1
  local meanstd
  if string.find(option.model.pretrainPath, 'vgg') then
    meanstd = {mean = torch[self.config.inputType]({123.68, 116.779, 103.939}),
               std = torch[self.config.inputType]({1, 1, 1})}
  else
    meanstd = {mean = torch[self.config.inputType]({0, 0, 0}),
               std = torch[self.config.inputType]({1, 1, 1})}
  end
  local norm = {mean = torch[self.config.inputType]({0, 0, 0}),
                std = torch[self.config.inputType]({255, 255, 255})}
  if self.config.preprocess == 1 then
    if split == 'train' then
      transforms = utility.img2d.transInOrder(
        {
          utility.img2d.cropRandomScale(224, 224),
         utility.img2d.normalize(meanstd),
         utility.img2d.normalize(norm),
         -- utility.img2d.scale(224, 224),
         -- utility.img2d.colorJitter(0.4, 0.4, 0.4),
         utility.img2d.hflip(0.5)})
    elseif split == 'val' then
      transforms = utility.img2d.transInOrder(
        {
          utility.img2d.normalize(meanstd),
          utility.img2d.normalize(norm),
          utility.img2d.scale(224, 224)
      })
    else
      error('invalid split: ' .. self.split)
    end
  elseif self.config.preprocess == 0 then
    transforms = utility.img2d.scale(224, 224)
  else
    error('not supported type of preprocessing')
  end
  
  return transforms(input, target)
end

