
local DatasetClsCIFAR, parent = torch.class('dataset.DatasetClsCIFAR', 'dataset.Dataset')

function DatasetClsCIFAR:__init(split, config)
  assert(split == 'train' or split == 'val', 'split must be train or val')
  if io.open(config.path['cacheFile'], 'r') == nil then
    local cacheMaker = dataset.CacheClsCIFAR(config)
    cacheMaker:save()
  end
  local cache = torch.load(config.path['cacheFile'])
  self.set = cache[split]
  self.clses = cache.classes
  self.lbls = cache.labels
  self.config = config
  self.split = split
  self.nChannel = 3
end

function DatasetClsCIFAR:get(i)
  return {input = self.set.data[i]:float(),
          target = self.set.label[i]}
end

function DatasetClsCIFAR:size()
  return self.set.data:size(1)
end

function DatasetClsCIFAR:histLabels()
  if self.hist == nil then
    self.hist = self.set.label:double():histc(#self.lbls)
  end
  return self.hist
end

function DatasetClsCIFAR:imageSize()
  return {self.nChannel, 32, 32} -- change the size according to the output of preprocess
end

function DatasetClsCIFAR:preprocess(sample, option, split)
  split = split or self.split
  local input, target, transforms = sample.input, sample.target
  local meanstd
  if self.config.nClass == 10 then
    -- compute from entire CIFAR-10 training set
    meanstd = {mean = torch.Tensor({125.3, 123.0, 113.9}),
               std = torch.Tensor({63.0, 62.1, 66.7})}
  elseif self.config.nClass == 100 then
    -- compute from entire CIFAR-100 training set with this code:
    -- dataset = torch.load('cifar-100.t7')
    -- tt = dataset.train.data:double():transpose(2,4):reshape(50000*32*32, 3)
    -- tt:mean(1)
    -- tt:std(1)
    meanstd = {mean = torch.Tensor({129.3, 124.1, 112.4}),
               std = torch.Tensor({68.2, 65.4, 70.4})}
  else
    error('invalid config.nClass (only 10 and 100): ' .. self.config.nClass)
  end
  if self.config.preprocess == 1 then
    if split == 'train' then
      transforms = utility.img2d.transInOrder(
        {utility.img2d.normalize(meanstd),
         utility.img2d.hflip(0.5),
         utility.img2d.cropRandomPad(32, 32, 4, 4)})
    elseif split == 'val' then
      transforms = utility.img2d.normalize(meanstd)
    else
      error('invalid split: ' .. self.split)
    end
    input = transforms(input)
  elseif self.config.preprocess == 0 then
  else
    error('not supported type of preprocessing')
  end
  return input, target
end

