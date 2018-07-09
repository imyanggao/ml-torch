
local DatasetSegIBSR, parent = torch.class('dataset.DatasetSegIBSR', 'dataset.Dataset')

function DatasetSegIBSR:__init(split, config)
  if io.open(config.path['cacheFile'], 'r') == nil then
    local cacheMaker = dataset.CacheSegIBSR(config)
    cacheMaker:save()
  end
  local cache = torch.load(config.path['cacheFile'])
  -- nSubject * nModality (T1, LM) * slice * height * width
  local allset = cache.cache
  -- if split is used, use config.trainProp to split train and val dataset
  if split == 'train' or split == 'val' then
    local nTrainSubject = math.ceil(allset:size(1) * config.trainProp)
    if split == 'train' then
      self.set = allset[{{1, nTrainSubject}}]
    elseif split == 'val' then
      self.set = allset[{{nTrainSubject + 1, allset:size(1)}}]
    end
  else
    self.set = allset
  end
  -- deal with 2D or 3D
  if config.dim == 2 then
    -- switch order of dimension to: nSubject * slice * nModality * height * width
    self.set =  self.set:transpose(2,3)
    -- merge the subject and slice dimension: nSubjectSlice * nModality * h * w
    local sz = {[1] = -1}
    for i = 3, self.set:dim() do
      sz[i-1] = self.set:size(i)
    end
    self.set = self.set:contiguous():view(table.unpack(sz))
    self.nChannel = self.set:size(self.set:dim()-2) - 1
  elseif config.dim == 3 then
    -- nothing need to do if 3D: nSubject * nModality * slice * h * w
    self.nSlice = self.set:size(self.set:dim()-2)
    self.nChannel = self.set:size(self.set:dim()-3) - 1
  else
    error('config.dim must be 2 or 3')
  end
  self.clses = cache.classes
  self.lbls = cache.labels
  self.split = split
  self.height, self.width = utility.img2d.size(self.set)
  self.config = config
end

function DatasetSegIBSR:get(i)
  return {input = self.set[{{i},{1}}]:squeeze(1),
          target = self.set[{{i},{2}}]:squeeze(2):squeeze(1)}
end

function DatasetSegIBSR:size()
  return self.set:size(1)
end

function DatasetSegIBSR:histLabels()
  if self.hist == nil then
    self.hist = self.set[{{},{-1}}]:double():histc(#self.lbls)
  end
  return self.hist
end

function DatasetSegIBSR:imageSize()
  return {self.nChannel, 224, 224} -- change the size according to the output of preprocess
end

function DatasetSegIBSR:preprocess(sample, option, split)
  split = split or self.split
  local paddedHeight, paddedWidth = 256, 256
  local height, width, transforms = 224, 224
  local hPad = math.ceil((paddedHeight - self.height) / 2)
  local wPad = math.ceil((paddedWidth - self.width) / 2)
  local targetPad = 1           -- background label
  if self.config.preprocess == 1 then
    if split == 'train' then
      transforms = utility.img2d.transInOrder(
        {utility.img2d.cropRandomPad(paddedHeight, paddedWidth, hPad, wPad, targetPad),
         utility.img2d.cropRandomScale(height, width),
         utility.img2d.hflip(0.5)})
    elseif split == 'val' then
      transforms = utility.img2d.transInOrder(
        {utility.img2d.cropRandomPad(paddedHeight, paddedWidth, hPad, wPad, targetPad),
         utility.img2d.scale(height, width)})
    else
      error('invalid split: ' .. self.split)
    end
  elseif self.config.preprocess == 0 then
    transforms = utility.img2d.transInOrder({
        utility.img2d.cropRandomPad(paddedHeight, paddedWidth, hPad, wPad, targetPad),
        utility.img2d.scale(height, width)})
  else
    error('not supported type of preprocessing')
  end
  
  return transforms(sample.input, sample.target)
end
