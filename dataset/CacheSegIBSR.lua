
local ffi = require 'ffi'

local CacheSegIBSR, parent = torch.class('dataset.CacheSegIBSR', 'dataset.Cache')

local function labelMap()
  local classes = {[0] = 'bg', [85] = 'csf', [170] = 'gray', [255] = 'white'}
  local label2class = {[1] = 0, [2] = 85, [3] = 170, [4] = 255}
  local class2label= {[0] = 1, [85] = 2, [170] = 3, [255] = 4}
  local labels = {}
  for class, name in pairs(classes) do
    labels[class2label[class]] = name
  end
  return labels, classes, label2class, class2label
end

local function rename(opt)
  local cachePath = utility.io.splitPath(opt.path['cacheFile']) 
  local renameRoot = paths.concat(cachePath, 'rename')
  utility.sys.mkdir(renameRoot)
  local originNames = utility.io.findByExts(opt.path['originDir'], opt.ext)
  local idxSub, subject, nameMap = {}, {}, {}
  local name, subjectStr, typeStr, newName
  for i = 1, #originNames do
    local path, basename, _ = utility.io.splitPath(originNames[i])
    name = paths.concat(path, basename) .. '.'
    subjectStr = string.format('%02d', name:match('IBSR_(%d+)_'))
    if idxSub[subjectStr] == nil then
      idxSub[subjectStr] = utility.tbl.len(idxSub) + 1
      subject[idxSub[subjectStr]] = subjectStr
    end
    if string.find(name, '_ana_strip') then typeStr = 'T1'
    elseif string.find(name, '_segTRI_ana') then typeStr = 'LM'
    else error("could not figure out type") end
    newName = renameRoot .. '/S' .. subjectStr .. '-' .. typeStr .. '.'
    utility.sys.exec('cp ' .. name .. opt.ext .. ' ' .. newName .. opt.ext)
    nameMap[newName .. opt.ext] = name .. opt.ext
  end
  return subject, nameMap
end

-- cached dataset: nSubject * nModality (T1, LM) * slice * height * width
local function read(opt, subject, nameMap)
  local cachePath = utility.io.splitPath(opt.path['cacheFile']) 
  local renameRoot = paths.concat(cachePath, 'rename')
  local loader3d, nameMapNew, cached, old, new = utility.img3d.load(cachePath), {}
  for i = 1, #subject do
    local oldName = renameRoot .. '/S' .. subject[i] .. '-'
    local newName = renameRoot .. '/' .. string.format('%02d', i) .. '-'
    for k, v in ipairs({'T1', 'LM'}) do
      old = oldName .. v .. '.'
      new = newName .. v .. '.'
      utility.sys.exec('mv ' .. old .. opt.ext .. ' ' .. new .. opt.ext)
      nameMapNew[string.format('%02d', i) .. '-' .. v] = nameMap[old .. opt.ext]
      local img3d = loader3d(new .. opt.ext)
      if i == 1 and v == 'T1' then
        cached = torch.ByteTensor(#subject, 2, table.unpack(img3d:size():totable()))
      end
      cached[i][k] = img3d
      print(' | ' .. string.format('%02d', i) .. '-' .. v)
    end
  end
  return cached, nameMapNew
end

-- apply label mapping to transfer label from 1 to #class
local function applyLabelMap(cached, labelMapping)
  local labelTensor = cached[{{},{2},{},{},{}}]
  local mapping = utility.img.transLabels(labelMapping)
  cached[{{},{2},{},{},{}}] = mapping(labelTensor)
  collectgarbage()
  return cached
end

function CacheSegIBSR:exec()
  if self.opt.path['originDir'] == nil or self.opt.ext == nil then
    error('need to specify opt: path[originDir] path[cacheFile] and ext')
  else
    print('=> Cache IBSR dataset ... ')
    print(self.opt)
  end
  print(' | load label mapping')
  local labels, classes, label2class, class2label = labelMap()
  print(' | rename 3D images')
  local subject, tmpMap = rename(self.opt)
  print(' | cache 3D images')
  local cached, nameMap = read(self.opt, subject, tmpMap)
  print(' | apply label mapping')
  cached = applyLabelMap(cached, class2label)
  self.dataset = {cache = cached, labels = labels, classes = classes,
                  label2class = label2class, class2label = class2label,
                  subject = subject, nameMap = nameMap}
end
