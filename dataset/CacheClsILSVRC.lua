
local ffi = require 'ffi'

local CacheClsILSVRC, parent = torch.class('dataset.CacheClsILSVRC', 'dataset.Cache')

local function parseMap(mapFile)
  local file = assert(io.open(mapFile, 'r'))
  local classes = {}
  local wnid = {}
  for line in file:lines() do
    local tbl = line:split(' ')
    local label = tonumber(tbl[2])
    wnid[tbl[1]] = label
    classes[label] = tbl[3]
  end
  return classes, wnid
end

local function parseTrain(trainFile, wnid)
  local file = assert(io.open(trainFile, 'r'))
  local tbl_path = {}
  local tbl_label = {}
  local maxStrLen = -1
  for line in file:lines() do
    local tbl1 = line:split(' ')
    local i = tonumber(tbl1[2])
    tbl_path[i] = tbl1[1]
    -- all bytes of the string plus a zero-terminator is required for ffi.copy
    maxStrLen = math.max(maxStrLen, #tbl1[1] + 1)
    local tbl2 = tbl1[1]:split('/')
    tbl_label[i] = wnid[tbl2[1]]
  end
  
  -- convert table to tensor for fast loading
  local nSample = #tbl_path
  local path = torch.CharTensor(nSample, maxStrLen):zero()
  for i, p in ipairs(tbl_path) do
    ffi.copy(path[i]:data(), p)
  end
  local label = torch.LongTensor(tbl_label)
  
  return {datapath = path, label = label}
end

local function parseVal(valFile, valTruthFile, valBlackListFile)
  local blacklist
  if valBlackListFile ~= nil then
    blacklist = {}
    local file = assert(io.open(valBlackListFile, 'r'))
    for line in file:lines() do
      blacklist[tonumber(line)] = 1
    end
  end
  local tbl_path = {}
  local maxStrLen = -1
  local i = 0
  local file = assert(io.open(valFile, 'r'))
  for line in file:lines() do
    local tbl = line:split(' ')
    local idx = tonumber(tbl[2])
    if blacklist == nil or blacklist[idx] ~= 1 then
      i = i + 1
      tbl_path[i] = tbl[1]
      -- all bytes of the string plus a zero-terminator is required for ffi.copy
      maxStrLen = math.max(maxStrLen, #tbl[1] + 1)
    end
  end
  local tbl_label = {}
  local i = 0
  local idx = 0
  local file = assert(io.open(valTruthFile, 'r'))
  for line in file:lines() do
    idx = idx + 1
    if blacklist == nil or blacklist[idx] ~= 1 then
      i = i + 1
      tbl_label[i] = tonumber(line)
    end
  end
  assert(#tbl_path == #tbl_label)

  -- convert table to tensor for fast loading
  local nSample = #tbl_path
  local path = torch.CharTensor(nSample, maxStrLen):zero()
  for i, p in ipairs(tbl_path) do
    ffi.copy(path[i]:data(), p)
  end
  local label = torch.LongTensor(tbl_label)
  
  return {datapath = path, label = label}
end

function CacheClsILSVRC:exec()
  if self.opt.path['mapFile'] == nil or self.opt.path['trainFile'] == nil
    or self.opt.path['valFile'] == nil or self.opt.path['valTruthFile'] == nil
    or self.opt.path['valBlackListFile'] == nil then
    error('need to specify opt paths: mapFile, trainFile, valFile, valTruthFile, valBlackListFile')
  else
    print('=> Cache ILSVRC-CLS dataset ... ')
    print(self.opt)
  end
  print(" | parsing map (classes) from " .. self.opt.path['mapFile'])
  local classes, wnid = parseMap(self.opt.path['mapFile'])
  print(" | parsing train from " .. self.opt.path['trainFile'])
  local train = parseTrain(self.opt.path['trainFile'], wnid)
  print(" | parsing test from " .. self.opt.path['valFile'])
  print(' | with truth from ' .. self.opt.path['valTruthFile'])
  print(' | and blacklist from ' .. self.opt.path['valBlackListFile'])
  local val = parseVal(self.opt.path['valFile'], self.opt.path['valTruthFile'], self.opt.path['valBlackListFile'])
  self.dataset = {train = train, val = val, labels = classes, classes = classes, wnid = wnid}
end
