
local img = {}

-- labelMapping is the table with (key = origin label, value = desired label)
function img.transLabels(labelMapping)
  return function(input)
    assert(input:type() == 'torch.ByteTensor', 'input must be torch.ByteTensor')
    local iLabels, oLabels, nLabel = {}, {}, 0
    for k, v in pairs(labelMapping) do
      nLabel = nLabel + 1
      iLabels[nLabel] = k
      oLabels[nLabel] = v
    end
    local output = - input - 1
    for i = 1, nLabel do
      output[output:eq(-iLabels[i]-1)] = oLabels[i]
    end
    return output:typeAs(input)
  end
end

function img.fft(dim)
  local signal = require('signal')
  local fft, ifft
  if dim == 1 then
    fft = signal.fft
    ifft = signal.ifft
  elseif dim == 2 then
    fft = signal.fft2
    ifft = signal.ifft2
  elseif dim == 3 then
    fft = signal.fft3
    ifft = signal.ifft3
  end
  return fft, ifft
end

-- image interpolation to return target image based on
-- input: source image I and inverse trainsfermation position grid
-- for 2D: I (height * width), grid (2 * height * width)
-- for 3D: I (slice * height * width), grid (3 * slice * height * width)
-- with different boundary wrapping: 'flip' (default) or 'truncate'
-- with different method: 'linear' (default) or 'cubic'
-- adapte with different image dimension with bilinear, trilinear, etc.
function img.interp(I, grid, wrap, method)
  wrap = wrap or 'truncate'
  method = method or 'linear'
  assert(grid:ne(grid):sum() == 0, 'input grid has nan, need to check reason')
  local dim, num, size = I:dim(), I:numel(), I:size()
  -- [1] is floor grid, [2] is ceil grid, [3] is (grid-floor)/(ceil-floor)
  local roundGrid = torch.Tensor(2, table.unpack(grid:size():totable()))
  torch.floor(roundGrid[1], grid)
  torch.ceil(roundGrid[2], grid)
  for i = 1, dim do
    if wrap == 'truncate' then
      roundGrid[{{1,2},{i}}][roundGrid[{{1,2},{i}}]:lt(1)] = 1
      roundGrid[{{1,2},{i}}][roundGrid[{{1,2},{i}}]:gt(size[i])] = size[i]
    elseif wrap == 'flip' then
      local wrapN = 0
      while roundGrid[{{1,2},{i}}]:lt(1):sum() ~= 0 do
        wrapN = wrapN + 1
        roundGrid[{{1,2},{i}}][roundGrid[{{1,2},{i}}]:lt(1)] =
          roundGrid[{{1,2},{i}}][roundGrid[{{1,2},{i}}]:lt(1)] + size[i]
      end
      if wrapN > 1 then print('warning: lower warpping more than once') end
      wrapN = 0
      while roundGrid[{{1,2},{i}}]:gt(size[i]):sum() ~= 0 do
        wrapN =  wrapN + 1
        roundGrid[{{1,2},{i}}][roundGrid[{{1,2},{i}}]:gt(size[i])] =
          roundGrid[{{1,2},{i}}][roundGrid[{{1,2},{i}}]:gt(size[i])] - size[i]
      end
      if wrapN > 1 then print('warning: upper warpping more than once') end
    else
      error('only truncate and flip are supported')
    end
  end
  local result
  if method == 'linear' then
    local d = torch.cdiv(grid - roundGrid[1], roundGrid[2] - roundGrid[1])
    d[roundGrid[1]:eq(roundGrid[2])] = 0.5
    local m = torch.Tensor(num, table.unpack(torch.totable(torch.ones(dim)*2)))
    if dim == 2 then
      local a = torch.Tensor(num, 1, 2)
      local b = torch.Tensor(num, 2, 1)
      for i = 1, 2 do
        for j = 1, 2 do
          local idx = (roundGrid[i][1] - 1) * size[2] + roundGrid[j][2]
          m[{{},{i},{j}}] = I:contiguous():view(-1):index(1, idx:view(-1):long())
        end
        if i == 1 then
          a[{{},{1},{i}}] = 1 - d[1]
          b[{{},{i},{1}}] = 1 - d[2]
        else
          a[{{},{1},{i}}] = d[1]
          b[{{},{i},{1}}] = d[2]
        end
      end
      result = torch.bmm(torch.bmm(a, m), b)
    elseif dim == 3 then
      local a = torch.Tensor(num, 1, 2)
      local b = torch.Tensor(num, 1, 2)
      local c = torch.Tensor(num, 2, 1)
      for i = 1, 2 do
        for j = 1, 2 do
          for k = 1, 2 do
            local idx = ((roundGrid[i][1] - 1) * size[3] + roundGrid[j][2] - 1) * size[2] + roundGrid[k][3]
            m[{{},{i},{j},{k}}] = I:contiguous():view(-1):index(1, idx:view(-1):long())
          end
        end
        if i == 1 then
          a[{{},{1},{i}}] = 1 - d[1]
          b[{{},{1},{i}}] = 1 - d[2]
          c[{{},{i},{1}}] = 1 - d[3]
        else
          a[{{},{1},{i}}] = d[1]
          b[{{},{1},{i}}] = d[2]
          c[{{},{i},{1}}] = d[3]
        end
      end
      result = torch.bmm(torch.bmm(b, torch.bmm(a, m:view(num, 2, -1)):view(num, 2, -1)), c)
    else
      error('only 2D and 3D are supported')
    end
  elseif method == 'cubic' then
  else
    error('only linear and cubic are supported')
  end
  return result:view(table.unpack(size:totable()))
end

utility.img = img
