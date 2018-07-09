
local image = require 'image'
require 'utility.init'

local DatasetSegSimple, parent = torch.class('dataset.DatasetSegSimple', 'dataset.Dataset')

function DatasetSegSimple:__init(split, config)
  self.split = split
  self.config = config
  self.rootData = config.root[split].data
  self.rootLabel = config.root[split].label
  self.namesData = utility.io.findByExts(self.rootData, config.exts)
  self.namesLabel = utility.io.findByExts(self.rootLabel, config.exts)
  self.sampleSz = #self.namesData
  self.cls = config.class
  assert(#self.namesData == #self.namesLabel,
         'the image count of data and label is NOT equal')
  if config.colormap ~= nil then
    self.Colormap = utility.Colormap(config.colormap)
  elseif config.labelMapping ~= nil then
    local nLabel = utility.tbl.len(config.labelMapping)
    local tmp = torch.Tensor(nLabel):fill(0)
    for k, v in pairs(config.labelMapping) do
      assert(v <= nLabel and v >= 1, 'config.labelMapping table value must be 1 to the number of class')
      tmp[v] = 1
    end
    assert(tmp:sum() == nLabel, 'config.labelMapping table value must be 1 to the number of class')
    self.labelMapping = config.labelMapping
  else
    error('if label image is color/gray, config.colormap/labelMapping need to be specified')
  end
end

function DatasetSegSimple:loadImage(path, type)
  local ok, img = pcall(function() return image.load(path, self.config.nChannel[type], self.config.tensorType[type]) end)
  if not ok then
    error('error reading: ' .. path)
  end
  return img
end

function DatasetSegSimple:get(i)
  local data = self:loadImage(self.namesData[i], 'data')
  local label = self:loadImage(self.namesLabel[i], 'label')
  if self.ColorMap ~= nil then
    label = self.Colormap:c2l(label)
  elseif self.labelMapping ~= nil then
    label = utility.img.transLabels(self.labelMapping)(label)
  end
  data, label = utility.img2d.scale(self.config.height, self.config.width)(data, label)
  label = label:squeeze(1)
  return {input = data, target = label}
end

function DatasetSegSimple:size()
  return #self.namesData
end

function DatasetSegSimple:histLabels()
  if self.hist == nil then
    print('Calculate ' .. self.split .. ' labels, this may take some time')
    local labels
    for i = 1, self:size() do
      xlua.progress(i, self:size())
      local _, label = self:preprocess(self:get(i))
      if i == 1 then
        labels = torch.Tensor(self:size(), table.unpack(label:size():totable()))
      end
      labels[i] = label
    end
    self.hist = labels:double():histc(#self.lbls)
  end
  return self.hist
end

function DatasetSegSimple:imageSize()
  return {self.config.nChannel, 224, 224} -- change the size according to the output of preprocess
end

function DatasetSegSimple:preprocess(sample, option, split)
  if self.config.preprocess == 1 then
    local height, width = 224, 224
    return utility.img2d.scale(height, width)(sample.input, sample.target)
  elseif self.config.preprocess == 0 then
    return sample.input, sample.target
  else
    error('not supported type of preprocessing')
  end
end
