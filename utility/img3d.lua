require('image')

local img3d = {}

-- size: slice * height * width
function img3d.info(name)
  local strInfo = utility.sys.exec('c3d ' .. name .. ' -info')
  local size, minmax = {}
  string.format('%d %d %d', strInfo:match('dim = %[(%d+), (%d+), (%d+)%]'))
  :gsub('%d+', function(n) size[3 - utility.tbl.len(size)] = tonumber(n) end)
  minmax = string.format('%s %s', strInfo:match('range = %[(%S+), (%S+)%]')):split(' ')
  for i = 1, #minmax do
    minmax[i] = tonumber(minmax[i])
  end
  return size, table.unpack(minmax)
end

-- boundary[1,2,3] = slice, height, width
function img3d.boundary(image3d)
  local function nonZero(tsr1D)
    local bStart, bEnd
    for i = 1, tsr1D:numel() do
      if tsr1D[i] ~= 0 then
        bStart = i
        break
      end
    end
    bEnd = bStart + tsr1D:ne(0):sum() - 1
    return {bStart, bEnd}
  end
  local boundary = {}
  boundary[1] = nonZero(image3d:sum(2):sum(3):squeeze())
  boundary[2] = nonZero(image3d:sum(1):sum(3):squeeze())
  boundary[3] = nonZero(image3d:sum(1):sum(2):squeeze())
  return boundary
end

-- load 3D image with 1 channel and byte format
-- byte because c3d (uchar, ushort) and image.load (byte, float, double) support
-- optional gMin, gMax is used to stretch intensity from [gMin, gMax]
-- otherwise intensity will be stretch from [min, max] of the input 3D image
function img3d.load(cacheRootDir, gMin, gMax)
  if gMin ~= nil then
    if gMax == nil then
      error('gMin and gMax must be all specified')
    end
  end
  return function(name)
    local size, min, max = img3d.info(name)
    min = gMin or min
    max = gMax or max
    local _, base, _ = utility.io.splitPath(name)
    local cacheDir = paths.concat(cacheRootDir, base)
    utility.sys.rm(cacheDir)
    utility.sys.mkdir(cacheDir)
    -- bugs in c3d or dataset, to get correct, need to change type and stretch
    local normalizedName = paths.concat(cacheDir, 'normalized.nii')
    utility.sys.exec('c3d ' .. name .. ' -stretch ' .. min .. ' ' .. max
                       .. ' 0 255 -type uchar -o ' .. normalizedName)
    -- with the normalized 3D image with uchar type, the following works
    -- c3d idx is 0-based
    utility.sys.exec('c3d ' .. normalizedName
                       .. ' -slice z 0:-1 -type uchar -oo '
                       .. paths.concat(cacheDir, '%03d.png'))
    local image3d, slicePath = torch.ByteTensor(table.unpack(size))    
    for i = 1, size[1] do
      -- slice name is 0-based because it from c3d
      slicePath = paths.concat(cacheDir, string.format('%03d.png', i - 1))
      -- lua idx is 1-based
      image3d[i] = image.load(slicePath, 1, 'byte')
    end
    collectgarbage()
    return image3d
  end
end

-- multi-thread version, just slightly fast if #slice ~ 200
function img3d.mtload(cacheRootDir, gMin, gMax)
  if gMin ~= nil then
    if gMax == nil then
      error('gMin and gMax must be all specified')
    end
  end
  return function(name)
    local size, min, max = img3d.info(name)
    min = gMin or min
    max = gMax or max
    local _, base, _ = utility.io.splitPath(name)
    local cacheDir = paths.concat(cacheRootDir, base)
    utility.sys.rm(cacheDir)
    utility.sys.mkdir(cacheDir)
    -- bugs in c3d or dataset, to get correct, need to change type and stretch
    local normalizedName = paths.concat(cacheDir, 'normalized.nii')
    utility.sys.exec('c3d ' .. name .. ' -stretch ' .. min .. ' ' .. max
                       .. ' 0 255 -type uchar -o ' .. normalizedName)
    -- with the normalized 3D image with uchar type, the following works
    -- c3d idx is 0-based
    utility.sys.exec('c3d ' .. normalizedName
                       .. ' -slice z 0:-1 -type uchar -oo '
                       .. paths.concat(cacheDir, '%03d.png'))
    local threads = require('threads')
    threads.Threads.serialization('threads.sharedserialize')
    local pool = threads.Threads(
      8,
      function()
        require('image')
      end
    )
    local image3d, i, j, batchSz = torch.ByteTensor(table.unpack(size)), 1, 1, 16
    local function queue()
      while i <= size[1] and pool:acceptsjob() do
        local curBatchSz = math.min(batchSz, size[1] - i + 1)
        pool:addjob(
          function(i, curBatchSz)
            for index = i, i + curBatchSz - 1 do
              -- slice name is 0-based because it from c3d, lua idx is 1-based
              local slicePath = paths.concat(
                cacheDir, string.format('%03d.png', index - 1))
              local slice = image.load(slicePath, 1, 'byte')
              if index == i then
                batch = torch.ByteTensor(curBatchSz, table.unpack(slice:squeeze():size():totable()))
              end
              batch[index - i + 1] = slice
            end
            return batch, i, curBatchSz
          end,
          function(batch, i, curBatchSz)
            image3d[{{i, i + curBatchSz - 1}}] = batch
          end,
          i, curBatchSz
        )
        i = i + curBatchSz
      end
      if pool:hasjob() then
        pool:dojob()
        return true
      end
    end
    repeat
    until not queue()
    pool:terminate()
    collectgarbage()
    return image3d
  end
end

utility.img3d = img3d
