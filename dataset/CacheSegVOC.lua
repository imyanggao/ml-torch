
local ffi = require 'ffi'

local CacheSegVOC, parent = torch.class('dataset.CacheSegVOC', 'dataset.Cache')

local function parseMap()
  local classes = {[0] = 'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    [255] = 'void'}
  local label2class = torch.range(0,21):totable()
  label2class[22] = 255
  local class2label = utility.tbl.inv(label2class)
  local labels = {}
  for class, name in pairs(classes) do
    labels[class2label[class]] = name
  end
  return labels, classes, label2class, class2label
end

local function parse(setFile)
  local file = assert(io.open(setFile, 'r'))
  local tbl_path = {}
  local maxStrLen = -1
  for line in file:lines() do
    table.insert(tbl_path, line)
    -- all bytes of the string plus a zero-terminator is required for ffi.copy
    maxStrLen = math.max(maxStrLen, #line + 1)
  end
  -- convert table to tensor for fast loading
  local nSample = #tbl_path
  local path = torch.CharTensor(nSample, maxStrLen):zero()
  for i, p in ipairs(tbl_path) do
    ffi.copy(path[i]:data(), p)
  end
  return {datapath = path, labelpath = path}
end

function CacheSegVOC:exec()
  if self.opt.path['trainFile'] == nil or self.opt.path['valFile'] == nil then
    error('need to specify opt: path[trainFile], path[valFile]')
  else
    print('=> Cache VOC-CLS-SEG dataset ... ')
    print(self.opt)
  end
  print(" | parsing map (class)")
  local labels, classes, label2class, class2label = parseMap()
  print(" | parsing train from " .. self.opt.path['trainFile'])
  local train = parse(self.opt.path['trainFile'])
  print(" | parsing test from " .. self.opt.path['valFile'])
  local val = parse(self.opt.path['valFile'])
  self.dataset = {train = train, val = val,
                  labels = labels, classes = classes,
                  label2class = label2class, class2label = class2label}
end
