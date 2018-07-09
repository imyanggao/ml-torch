
local DatasetSegBRIC, parent = torch.class('dataset.DatasetSegBRIC', 'dataset.Dataset')

function DatasetSegBRIC:__init(split, config)
  if io.open(config.path['cacheFile'], 'r') == nil then
    local cacheMaker = dataset.CacheSegBRIC(config)
    cacheMaker:save()
  end
  local cache = torch.load(config.path['cacheFile'])
  -- nSubject * nTime * nModality (T1, T2, LA, LM) * slice * height * width
  local allset, allSliceLM = cache.cache, cache.sliceLM
  -- if split is used, use config.trainProp to split train and val dataset
  if split == 'train' or split == 'val' then
    local nTrainSubject = math.ceil(allset:size(1) * config.trainProp)
    if split == 'train' then
      self.set = allset[{{1, nTrainSubject}}]
      self.sliceLM = {table.unpack(allSliceLM, 1, nTrainSubject)}
    elseif split == 'val' then
      self.set = allset[{{nTrainSubject + 1, allset:size(1)}}]
      self.sliceLM = {table.unpack(allSliceLM, nTrainSubject + 1, allset:size(1))}
    end
  else
    self.set = allset
    self.sliceLM = allSliceLM
  end
  -- use manual label or auto label with config.label = 'LM' or 'LA'
  if config.label == 'LM' then
    -- nSubject * nTime * nModality(T1, T2, LM) * 1 * height * width
    local nSubject, nTime, _, _, height, width = table.unpack(self.set:size():totable())
    local set = torch.ByteTensor(nSubject, nTime, 3, 1, height, width)
    for i = 1, #self.sliceLM do
      for j = 1, #self.sliceLM[i] do
        set[i][j][1][1] = self.set[i][j][1][self.sliceLM[i][j]]
        set[i][j][2][1] = self.set[i][j][2][self.sliceLM[i][j]]
        set[i][j][3][1] = self.set[i][j][4][self.sliceLM[i][j]]
      end
    end
    self.set = set
  elseif config.label == 'LA' then
    -- nSubject * nTime * nModality(T1, T2, LA) * slice * height * width
    self.set = self.set[{{},{},{1,3},{},{},{}}]
  else
    error("config.label must be 'LM' or 'LA'")
  end
  -- pick one modality or all modalities
  if config.modality == 'T1' then
    self.set = torch.cat(self.set[{{},{},{1}}], self.set[{{},{},{3}}], 3)
  elseif config.modality == 'T2' then
    self.set = torch.cat(self.set[{{},{},{2}}], self.set[{{},{},{3}}], 3)
  else
    -- nothing need to do
  end
  -- deal with 2D or 3D
  if config.dim == 2 then
    -- switch order of dimension to: nSubject * slice * nTime * nModality * height * width
    self.set =  self.set:transpose(3,4):transpose(2,3)
    -- merge the subject and slice dimension: nSubjectSlice * nTime * nModality * h * w
    local sz = {[1] = -1}
    for i = 3, self.set:dim() do
      sz[i-1] = self.set:size(i)
    end
    self.set = self.set:contiguous():view(table.unpack(sz))
    self.nChannel = self.set:size(self.set:dim()-2) - 1
  elseif config.dim == 3 then
    -- nothing need to do if 3D: nSubject * nTime * nModality * slice * h * w
    self.nSlice = self.set:size(self.set:dim()-2)
    self.nChannel = self.set:size(self.set:dim()-3) - 1
  else
    error('config.dim must be 2 or 3')
  end
  -- treat as longitudinal with config.time = 1
  if config.time == 1 then
    self.tm = cache.month
    -- switch time and subject dimension:
    -- either: nTime * nSubject * nModality * slice * height * width
    -- or:     nTime * nSubjectSlice * nModality * height * width
    self.set = self.set:transpose(1, 2)
  else
    self.tm = nil
    local sz = {[1] = -1}
    -- merge the subject and time dimension
    -- either: nSubjectTime * nModality * slice * height * width
    -- or:     nSubjectSliceTime * nModality * height * width
    for i = 3, self.set:dim() do
      sz[i-1] = self.set:size(i)
    end
    self.set = self.set:contiguous():view(table.unpack(sz))
  end

  self.clses = cache.classes
  self.lbls = cache.labels
  self.split = split
  self.height, self.width = utility.img2d.size(self.set)
  self.config = config
end

function DatasetSegBRIC:get(i)
  local input, target
  if self.tm ~= nil then
    input, target = self.set[{{},{i},{1,-2}}]:squeeze(2), self.set[{{},{i},{-1}}]:squeeze(3):squeeze(2)
  else
    input, target = self.set[{{i},{1,-2}}]:squeeze(1), self.set[{{i},{-1}}]:squeeze(2):squeeze(1)
  end
  return {input = input, target = target}
end

function DatasetSegBRIC:size()
  if self.tm ~= nil then
    return self.set:size(2)
  else
    return self.set:size(1)
  end
end

function DatasetSegBRIC:histLabels()
  if self.hist == nil then
    if self.tm ~= nil then
      self.hist = self.set[{{},{},{-1}}]:double():histc(#self.lbls)
    else
      self.hist = self.set[{{},{-1}}]:double():histc(#self.lbls)
    end
  end
  return self.hist
end

function DatasetSegBRIC:imageSize()
  return {self.nChannel, 224, 224} -- change the size according to the output of preprocess
end

function DatasetSegBRIC:preprocess(sample, option, split)
  split = split or self.split
  local height, width, transforms, input, target = 224, 224
  if self.config.preprocess == 1 then
    if split == 'train' then
      transforms = utility.img2d.transInOrder(
        {utility.img2d.cropRandomScale(height, width),
         utility.img2d.hflip(0.5)})
    elseif split == 'val' then
      transforms = utility.img2d.transInOrder(
        {utility.img2d.scale(height, width)})
    else
      error('invalid split: ' .. self.split)
    end
  elseif self.config.preprocess == 0 then
    transforms = utility.img2d.scale(height, width)
  else
    error('not supported type of preprocessing')
  end
  
  if self.tm ~= nil then
    local inputSz, targetSz = sample.input:size(), sample.target:size()
    inputSz[sample.input:dim()-1] = height
    inputSz[sample.input:dim()] = width
    targetSz[sample.target:dim()-1] = height
    targetSz[sample.target:dim()] = width
    input = torch.Tensor(inputSz):typeAs(sample.input)
    target = torch.Tensor(targetSz):typeAs(sample.target)
    for t = 1, #self.tm[1] do
      input[t], target[t] = transforms(sample.input[t], sample.target[t])
    end
  else
    input, target = transforms(sample.input, sample.target)
  end
  return input, target
end
