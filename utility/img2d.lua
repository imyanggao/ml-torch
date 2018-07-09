require 'image'

local img2d = {}

-- function img2d.scale(height, width, type)
--   type = type or 'data'
--   return function(input)
--     assert(input ~= nil and height ~= nil and width ~= nil,
--            'usage: img2d.scale(height, width, ...)(input) must be given')
--     local output
--     if type == 'data' then
--       output = image.scale(input, width, height)
--     elseif type == 'label' then
--       -- assert(input:type() == 'torch.ByteTensor', 'input must be torch.ByteTensor')
--       local labels = utility.tsr.unique(input)
--       local nLabel = #labels
--       output = torch.DoubleTensor(nLabel, height, width)
--       for i = 1, nLabel do
--         output[i] = image.scale(input:eq(labels[i]):double(), width, height)
--       end
--       _, output = output:max(1)
--       output = - output - 1
--       for i = 1, nLabel do
--         output[output:eq(-i-1)] = labels[i]
--       end
--     else
--       error('supported type: data or label')
--     end
--     return output:typeAs(input)
--   end
-- end

function img2d.linearMinMaxNormal()
  return function(input, target)
    if input ~= nil then
      local iType, iMin, iMax = input:type(), input:min(), input:max()
      input = (input:double() - iMin) / (iMax - iMin)
      if string.find(iType, 'Byte') ~= nil then
        input = input * 255
      end
      input = input:type(iType)
    end
    if target ~= nil then
      local tType, tMin, tMax = target:type(), target:min(), target:max()
      target = (target:double() - tMin) / (tMax - tMin)
      if string.find(tType, 'Byte') ~= nil then
        target = target * 255
      end
      target = target:type(tType)
    end
    return input, target
  end
end

function img2d.size(input, target)
  local height, width = 0, 0
  if input ~= nil then
    height, width = input:size(input:dim()-1), input:size(input:dim())
  elseif target ~= nil then
    height, width = target:size(target:dim()-1), target:size(target:dim())
  end
  return height, width
end

function img2d.transInOrder(transforms)
  return function(input, target)
    for _, transform in ipairs(transforms) do
      input, target = transform(input, target)
    end
    return input, target
  end
end

function img2d.transPermute(transforms)
  return function(input, target)
    local idx = torch.randperm(#transforms)
    for i = 1, #transforms do
      input, target = transforms[idx[i]](input, target)
    end
    return input, target
  end
end

function img2d.normalize(meanstd)
  local nChannel = meanstd.mean:nElement()
  return function(input, target)
    assert(input:size(1) == nChannel, 'nChannel is not the same between input image and meanstd')
    if input ~= nil then
      input = input:clone()
      for i = 1, nChannel do
        input[i]:add(-meanstd.mean[i])
        input[i]:div(meanstd.std[i])
      end
    end
    return input, target
  end
end

function img2d.hflip(prob)
  return function(input, target)
    if torch.uniform() < prob then
      if input ~= nil then
        input = image.hflip(input)
      end
      if target ~= nil then
        target = image.hflip(target)
      end
    end
    return input, target
  end
end

function img2d.rotate(degree, interpolation)
  interpolation = interpolation or 'bilinear'
  local radian = degree * math.pi / 180
  return function(input, target)
    local radianRandom = (torch.uniform() - 0.5) * radian
    if input ~= nil then
      input = image.rotate(input, radianRandom, interpolation)
    end
    if target ~= nil then
      local labels = utility.tsr.unique(target)
      local nLabel = #labels
      local height, width = img2d.size(target)
      local tmp = torch.DoubleTensor(nLabel, height, width)
      for i = 1, nLabel do
        tmp[i] = image.rotate(target:eq(labels[i]):double(), radianRandom, interpolation)
      end
      _, tmp = tmp:max(1) 
      tmp = tmp:squeeze(1)        -- may not need squeeze??
      tmp = - tmp - 1
      for i = 1, nLabel do
        tmp[tmp:eq(-i-1)] = labels[i]
      end
      target = tmp:typeAs(target)
    end
    return input, target
  end
end

-- AlexNet-style PCA-based noise
function img2d.lighting(alphastd, pca)
  local nChannel = pca.eigval:nElement()
  return function(input, target)
    if alphastd <= 0 then
      return input, target
    end
    assert(input:size(1) == nChannel, 'nChannel is not the same between input image and pca')
    local tmp = pca.eigvec:clone()
      :cmul(torch.Tensor(1, nChannel):normal(0, alphastd):expand(nChannel, nChannel))
      :cmul(pca.eigval:view(1, nChannel):expand(nChannel, nChannel))
      :sum(2):squeeze()
    input = input:clone()
    for i = 1, nChannel do
      input[i]:add(tmp[i])
    end
    return input, target
  end
end

local function scale(input, target, height, width, interpolation)
  interpolation = interpolation or 'bicubic'
  if input ~= nil then
    input = image.scale(input, width, height, interpolation)
  end
  if target ~= nil then
    local labels = utility.tsr.unique(target)
    local nLabel = #labels
    local tmp = torch.DoubleTensor(nLabel, height, width)
    for i = 1, nLabel do
      tmp[i] = image.scale(target:eq(labels[i]):double(), width, height, interpolation)
    end
    _, tmp = tmp:max(1) 
    tmp = tmp:squeeze(1)        -- may not need squeeze??
    tmp = - tmp - 1
    for i = 1, nLabel do
      tmp[tmp:eq(-i-1)] = labels[i]
    end
    target = tmp:typeAs(target)
  end
  return input, target
end

function img2d.scale(height, width, interpolation)  
  assert(height ~= nil and width ~= nil, 'height and width must be given')
  return function(input, target)
    return scale(input, target, height, width, interpolation)
  end
end

-- scale the short edge to size
function img2d.scaleShortEdge(size, interpolation)
  return function(input, target)
    local height, width = img2d.size(input, target)
    if width < height then
      return scale(input, target, torch.round(height / width * size), size, interpolation)
    else
      return scale(input, target, size, torch.round(width / height * size), interpolation)
    end
  end
end

-- ResNet-style: resized with shorter edge randomly sampled from [minSize, maxSize]
function img2d.scaleRandomShortEdge(minSize, maxSize, interpolation)
  return function(input, target)
    local height, width = img2d.size(input, target)
    local size = torch.random(minSize, maxSize)
    return img2d.scaleShortEdge(size, interpolation)(input, target)
  end
end

-- 1-based coordinate
local function crop(input, target, h1, w1, hSize, wSize)
  if input ~= nil then
    input = image.crop(input, w1 - 1, h1 - 1, w1 + wSize - 1, h1 + hSize - 1)
  end
  if target ~= nil then
    target = image.crop(target, w1 - 1, h1 - 1, w1 + wSize - 1, h1 + hSize - 1)
  end
  return input, target
end

function img2d.cropCenter(hSize, wSize)
  wSize = wSize or hSize
  return function(input, target)
    local height, width = img2d.size(input, target)
    local h1 = math.floor((height - hSize) / 2) + 1
    local w1 = math.floor((width - wSize) / 2) + 1
    return crop(input, target, h1, w1, hSize, wSize)
  end
end

-- random crop image with input padding zero and target padding targetPad
function img2d.cropRandomPad(hSize, wSize, hPad, wPad, targetPad)
  wSize  = wSize or hSize
  hPad = hPad or 0
  wPad = wPad or hPadp
  return function(input, target)
    if target ~= nil and hPad > 0 and targetPad == nil then
      error('with zero padding on input, target void padding value must be given at 5th parameter')
    end
    if hPad > 0 then
      if input ~= nil then
        local inputDims = input:size()
        inputDims[input:dim()-1] = inputDims[input:dim()-1] + 2 * hPad
        inputDims[input:dim()] = inputDims[input:dim()] + 2 * wPad
        local inputPad = torch.zeros(inputDims):typeAs(input)
        inputPad:narrow(input:dim()-1, hPad+1, input:size(input:dim()-1))
          :narrow(input:dim(), wPad+1, input:size(input:dim())):copy(input)
        input = inputPad
      end
      if target ~= nil then
        local targetDims = target:size()
        targetDims[target:dim()-1] = targetDims[target:dim()-1] + 2 * hPad
        targetDims[target:dim()] = targetDims[target:dim()] + 2 * wPad
        local targetPad = torch.Tensor(targetDims):fill(targetPad):typeAs(target)
        targetPad:narrow(target:dim()-1, hPad+1, target:size(target:dim()-1))
          :narrow(target:dim(), wPad+1, target:size(target:dim())):copy(target)
        target = targetPad
      end
    end
    local height, width = img2d.size(input, target)
    local h1, w1 = torch.random(1, height - hSize + 1), torch.random(1, width - wSize + 1)
    return crop(input, target, h1, w1, hSize, wSize)
  end
end

-- Inception-style: random crop with area 8% ~ 100% and aspect ratio 3/4 ~ 4/3
function img2d.cropRandomScale(hSize, wSize, interpolation)
  wSize = wSize or hSize
  local centerCrop = img2d.cropCenter(hSize, wSize)
  return function(input, target)
    local height, width = img2d.size(input, target)
    local areaOrigin = height * width
    local attempt = 0
    repeat
      local areaNew = torch.uniform(0.08, 1.0) * areaOrigin
      local ratioNew = torch.uniform(3/4, 4/3)
      local heightNew = torch.round(math.sqrt(areaNew * ratioNew))
      local widthNew = torch.round(math.sqrt(areaNew / ratioNew))
      if torch.uniform() < 0.5 then
        heightNew, widthNew = widthNew, heightNew
      end
      if heightNew <= height and widthNew <= width then
        local h1 = torch.random(1, height - heightNew + 1)
        local w1 = torch.random(1, width - widthNew + 1)
        input, target = crop(input, target, h1, w1, heightNew, widthNew)
        return scale(input, target, hSize, wSize, interpolation)
      end
      attempt = attempt + 1
    until attempt >= 10

    local ratio = height / width
    if ratio * wSize > hSize then
      return centerCrop(scale(input, target, ratio * wSize, wSize, interpolation))
    else
      return centerCrop(scale(input, target, hSize, hSize / ratio, interpolation))
    end
  end
end

-- 4 corners and center crop from image and its horizontal flip
function img2d.crop10(hSize, wSize)
  wSize = wSize or hSize
  local centerCrop = img2d.cropCenter(hSize, wSize)
  return function(input, target)
    local height, width = img2d.size(input, target)
    local h1 = {math.floor((height - hSize) / 2) + 1,
                1,
                1,
                height - hSize + 1,
                height - hSize + 1}
    local w1 = {math.floor((width - wSize) / 2) + 1,
                1,
                width - wSize + 1,
                1,
                width - wSize + 1}
    local cropInputs, cropTargets, cropInput, cropTarget = nil, nil
    if input ~= nil then
      local tmp = input:size():totable()
      tmp[input:dim()-1] = hSize
      tmp[input:dim()] = wSize
      cropInputs = torch.Tensor(table.unpack(utility.tbl.cat({2 * #h1}, tmp))):typeAs(input)
    end
    if target ~= nil then
      local tmp = target:size():totable()
      tmp[target:dim()-1] = hSize
      tmp[target:dim()] = wSize
      cropTargets = torch.Tensor(table.unpack(utility.tbl.cat({2 * #h1}, tmp))):typeAs(target)
    end
    for i = 1, #h1 do
      cropInput, cropTarget = crop(input, target, h1[i], w1[i], hSize, wSize)
      if cropInput ~= nil then
        cropInputs[(i-1)*2+1] = cropInput
      end
      if cropTarget ~= nil then
        cropTargets[(i-1)*2+1] = cropTarget
      end
      local hflipInput, hflipTarget = nil, nil
      if input ~= nil then
        hflipInput = image.hflip(input)
      end
      if target ~= nil then
        hflipTarget = image.hflip(target)
      end
      cropInput, cropTarget = crop(hflipInput, hflipTarget, h1[i], w1[i], hSize, wSize)
      if cropInput ~= nil then
        cropInputs[i*2] = cropInput
      end
      if cropTarget ~= nil then
        cropTargets[i*2] = cropTarget
      end
    end
    return cropInputs, cropTargets
  end
end

local function blend(img1, img2, w)
  return img1 * w + img2 * (1 - w)
end

local function gray(dst, img)
  dst:resizeAs(img)
  dst[1]:zero()
  dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
  dst[2]:copy(dst[1])
  dst[3]:copy(dst[1])
  return dst
end

function img2d.brightness(var)
  local cache
  return function(input, target)
    cache = cache or input.new()
    cache:resizeAs(input):zero()
    input = blend(input, cache, 1 + torch.uniform(-var, var))
    return input, target
  end
end

function img2d.contrast(var)
  local cache
  return function(input, target)
    cache = cache or input.new()
    gray(cache, input)
    cache:fill(cache[1]:mean())
    input = blend(input, cache, 1 + torch.uniform(-var, var))
    return input, target
  end
end

function img2d.saturation(var)
  local cache
  return function(input, target)
    cache = cache or input.new()
    gray(cache, input)
    input = blend(input, cache, 1 + torch.uniform(-var, var))
    return input, target
  end
end

function img2d.colorJitter(brightnessVar, contrastVar, saturationVar)
  brightnessVar = brightnessVar or 0
  contrastVar = contrastVar or 0
  saturationVar = saturationVar or 0
  local transforms = {}
  table.insert(transforms, img2d.brightness(brightnessVar))
  table.insert(transforms, img2d.contrast(contrastVar))
  table.insert(transforms, img2d.saturation(saturationVar))
  return img2d.transPermute(transforms)
end

utility.img2d = img2d
