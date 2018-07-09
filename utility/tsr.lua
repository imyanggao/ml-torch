
local tsr = {}

function tsr.unique(x, rank)
  if rank == nil then rank = true else rank = false end
  local y = x:view(-1)
  local unique = {}
  local exist = {}
  for i = 1, y:numel() do
    if exist[y[i]] == null then
      exist[y[i]] = true
      table.insert(unique, y[i])
    end
  end
  if rank then
    table.sort(unique)
  end
  return unique
end

function tsr.meshgrid(sizetbl)
  local dim = utility.tbl.len(sizetbl)
  local grid = torch.Tensor(dim, table.unpack(sizetbl))
  for i = 1, dim do
    local s = torch.range(1, sizetbl[i])
    local v = torch.ones(dim)
    v[i] = sizetbl[i]
    -- maybe a bug here: expand don't allocate new memory, need double check
    grid[i] = torch.expand(s:view(table.unpack(v:totable())),
                           table.unpack(sizetbl))
  end
  return grid
end

-- input x should be h * w for 2D and slice * h * w for 3D
-- output: 2 * h * w for 2D and 3 * slice * h * w for 3D
function tsr.grad(x)
  local grad = torch.Tensor(x:dim(), table.unpack(x:size():totable()))
    :zero():typeAs(x)
  for i = 1, x:dim() do
    local sz = x:size(i)
    -- take forward differences on left and right edges
    if sz > 1 then
      grad[i]:narrow(i, 1, 1)
        :copy(x:narrow(i, 2, 1) - x:narrow(i, 1, 1))
      grad[i]:narrow(i, sz, 1)
        :copy(x:narrow(i, sz, 1) - x:narrow(i, sz - 1, 1))
    end
    -- take centered differences on interior points
    if sz > 2 then
      grad[i]:narrow(i, 2, sz - 2)
        :copy((x:narrow(i, 3, sz - 2) - x:narrow(i, 1, sz - 2)) / 2)
    end
  end
  return grad
end

-- input x should be 2 * h * w for 2D and 3 * slice * h * w for 3D
-- output: 2 * 2 * h * w for 2D and 3 * 3 * slice * h * w for 3D
function tsr.jacob(x)
  local jacobian = torch.Tensor(x:size(1), table.unpack(x:size():totable()))
    :typeAs(x)
  for i = 1, x:size(1) do
    jacobian[i] = tsr.grad(x[i])
  end
  return jacobian
end

-- input x should be 2 * h * w for 2D and 3 * slice * h * w for 3D
-- output: h * w for 2D and slice * h * w for 3D
function tsr.div(x)
  local divergence = torch.zeros(
    table.unpack(torch.Tensor(x:size():totable())[{{2,-1}}]:totable()))
    :typeAs(x)
  for i = 1, x:size(1) do
    divergence = divergence + tsr.grad(x[i])[i]
  end
  return divergence
end

utility.tsr = tsr
