
local ffi = require 'ffi'
local image = require 'image'

local DatasetClsILSVRC, parent = torch.class('dataset.DatasetClsILSVRC', 'dataset.Dataset')

function DatasetClsILSVRC:__init(split, config)
  assert(split == 'train' or split == 'val' or split == 'test', 'split must be train, val or test') 
  if io.open(config.path['cacheFile'], 'r') == nil then
    local cacheMaker = dataset.CacheClsILSVRC(config)
    cacheMaker:save()
  end
  local cache = torch.load(config.path['cacheFile'])
  self.set = cache[split]
  self.clses = cache.classes
  self.lbls = cache.labels
  self.split = split
  if split == 'train' then
    self.rootData = config.path['trainDir']
  elseif split == 'test' then
    self.rootData = config.path['testDir']
  elseif split == 'val' then
    self.rootData = config.path['valDir']
  end
  self.config = config
  self.nChannel = 3
end

function DatasetClsILSVRC:loadImage(path, type)
  local ok, img = pcall(function() return image.load(path, self.nChannel, self.config.tensorType[type]) end)
  if not ok then
    error('error reading: ' .. path)
  end
  return img
end

function DatasetClsILSVRC:get(i)
  local path = ffi.string(self.set.datapath[i]:data())
  local data = self:loadImage(paths.concat(self.rootData, path) .. '.JPEG', 'data')
  local label = self.set.label[i]
  return {input = data:type('torch.' .. self.config.inputType),
          target = label}
end

function DatasetClsILSVRC:size()
  return self.set.datapath:size(1)
end

function DatasetClsILSVRC:histLabels()
  if self.hist == nil then
    self.hist = self.set.label:double():histc(#self.lbls)
  end
  return self.hist
end

function DatasetClsILSVRC:imageSize()
  return {self.nChannel, 224, 224} -- change the size according to the output of preprocess
end

function DatasetClsILSVRC:preprocess(sample, option, split)
  split = split or self.split
  local input, target, transforms = sample.input, sample.target
  -- compute from random subset of ImageNet training set
  local meanstd = {mean = torch[self.config.inputType]({0.485, 0.456, 0.406}),
                   std = torch[self.config.inputType]({0.229, 0.224, 0.225})}
  local pca = {eigval = torch[self.config.inputType]({0.2175, 0.0188, 0.0045}),
               eigvec = torch[self.config.inputType]({
                   {-0.5675, 0.7192, 0.4009},
                   {-0.5808, -0.0045, -0.8140},
                   {-0.5836, -0.6948, 0.4203}})}
  if self.config.preprocess == 1 then
    if split == 'train' then
      transforms = utility.img2d.transInOrder(
        {utility.img2d.cropRandomScale(224, 224),
         utility.img2d.colorJitter(0.4, 0.4, 0.4),
         utility.img2d.lighting(0.1, pca),
         utility.img2d.normalize(meanstd),
         utility.img2d.hflip(0.5)})
    elseif split == 'val' then
      transforms = utility.img2d.transInOrder(
        {utility.img2d.scaleShortEdge(256),
         utility.img2d.normalize(meanstd),
         utility.img2d.cropCenter(224, 224)})
    else
      error('invalid split: ' .. self.split)
    end
  elseif self.config.preprocess == 0 then
    transforms = utility.img2d.transInOrder({utility.img2d.scaleShortEdge(256),
                                             utility.img2d.cropCenter(224, 224)})
  else
    error('not supported type of preprocessing')
  end
  
  return transforms(input), target
end

