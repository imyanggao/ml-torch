
local GeodesicShooting = torch.class('utility.GeodesicShooting')

function GeodesicShooting:__init(sizetbl, alpha, gamma, lpower, sigma, t, lr)
  self.sizetbl = sizetbl
  self.alpha = alpha or 3  -- 0.3
  self.gamma = gamma or 0.1  -- 0.1
  self.lpower = lpower or 2
  self.sigma = sigma or 0.01 -- 0.03
  self.t = t or 10
  self.lr = lr or 1e-4
  self.dt = 1 / self.t
  -- self.v and self.phiinv is the tables contain v and phiinv respectively
  -- at each step in geodesic with
  -- 2 * height * width for 2D, 3 * slice * height * width for 3D
  self.v = {}
  self.phiinv = {}
  self.dim = utility.tbl.len(sizetbl)
  self.fft, self.ifft = utility.img.fft(self.dim)
  self:fftL()
  -- tmp: 2 * height * width for 2D, 3 * slice * height * width for 3D
  self.tmp1 = torch.Tensor(self.dim, table.unpack(sizetbl))
  self.tmp2 = self.tmp1:clone()
  self.tmp3 = self.tmp1:clone()
end

-- L, K: height * width for 2D, slice * height * width for 3D
function GeodesicShooting:fftL()
  self.meshgrid = utility.tsr.meshgrid(self.sizetbl)
  self.L = torch.zeros(table.unpack(self.sizetbl))
  for i = 1, self.dim do
    self.L = self.L + torch.cos(2 * math.pi * (self.meshgrid[i]-1) / self.sizetbl[i])
  end
  self.L = torch.pow(-2 * self.alpha * (self.L - 2) + self.gamma, self.lpower)
  self.K = torch.cinv(self.L)
end

-- F: height * width for 2D, slice * height * width for 3D
-- x: 2 * height * width for 2D, 3 * slice * height * width for 3D
function GeodesicShooting:apply(F, x)
  local sizetbl = F:size():totable()
  -- maybe a bug here: expand don't allocate new memory, need double check
  F = F:view(table.unpack(utility.tbl.cat(sizetbl, 1)))
    :expand(table.unpack(utility.tbl.cat(sizetbl, 2)))
  for i = 1, self.dim do
    self.tmp1[i]:copy(self.ifft(torch.cmul(self.fft(x[i]), F))
                        :select(1 + self.dim, 1))
  end
  return self.tmp1:clone()
end

-- x, y: 2 * height * width for 2D, 3 * slice * height * width for 3D
function GeodesicShooting:ad(x, y)
  local dx, dy = utility.tsr.jacob(x), utility.tsr.jacob(y)
  self.tmp1:zero()
  self.tmp2:zero()
  for i = 1, self.dim do
    for j = 1, self.dim do
      self.tmp1[i]:addcmul(1, dx[i][j], y[j])
      self.tmp2[i]:addcmul(1, dy[i][j], x[j])
    end
  end
  return self.tmp1 - self.tmp2
end

-- x, y: 2 * height * width for 2D, 3 * slice * height * width for 3D
function GeodesicShooting:adDagger(x, y)
  local Ly = self:apply(self.L, y)
  local dx = utility.tsr.jacob(x)
  self.tmp1:zero()
  for i = 1, self.dim do
    for j = 1, self.dim do
      self.tmp1[i]:addcmul(1, dx[j][i], Ly[j])
      torch.cmul(self.tmp2[j], Ly[i], x[j])
    end
    self.tmp3[i] = utility.tsr.div(self.tmp2)
  end
  return self:apply(self.K, self.tmp1 + self.tmp3)
end

function GeodesicShooting:forwardV()
  for i = 1, self.t do
    self.v[i] = self.v[i-1] - self:adDagger(self.v[i-1], self.v[i-1]) * self.dt
  end
end

function GeodesicShooting:cstrBoundary(jacob)
  for i = 1, self.dim do
    local size = self.sizetbl[i]
    jacob[i][i]:select(i, 1):add(size / 2)
    jacob[i][i]:select(i, size):add(size / 2)
  end
  return jacob
end

function GeodesicShooting:forwardPhiinv()
  self.tmp1 = self.phiinv[0]
  for i = 1, self.t do
    local d_tmp = self:cstrBoundary(utility.tsr.jacob(self.tmp1))
    self.tmp2:zero()
    for j = 1, self.dim do
      for k = 1, self.dim do
        self.tmp2[j]:addcmul(1, d_tmp[j][k], self.v[i-1][k])
      end
    end
    self.tmp1 = self.tmp1 - self.tmp2 * self.dt
    self.phiinv[i] = self.tmp1
  end
end

function GeodesicShooting:backwardV()
  self.dv0 = torch.zeros(self.dim, table.unpack(self.sizetbl))
  for i = self.t, 1, -1 do
    self.tmp1 = self:ad(self.v[i-1], self.dv0)
    self.tmp2 = self:adDagger(self.dv0, self.v[i-1])
    self.dv0 = self.dv0 - (self.tmp1 - self.tmp2 - self.dv1) * self.dt
    self.dv1 = self.dv1 + self:adDagger(self.v[i-1], self.dv1) * self.dt
  end
end

function GeodesicShooting:shoot(src, tgt, v0)
  self.v[0] = v0
  self:forwardV()
  self.phiinv[0] = utility.tsr.meshgrid(self.sizetbl)
  self:forwardPhiinv()
  local deform = utility.img.interp(src, self.phiinv[self.t])
  local d_deform = utility.tsr.grad(deform)
  self.dv1 = torch.zeros(self.dim, table.unpack(self.sizetbl))
  for i = 1, self.dim do
    self.dv1[i]:addcmul(-1, deform - tgt, d_deform[i])
  end
  self.dv1 = self:apply(self.K, self.dv1)
  self:backwardV()
  self.dv0 = self.dv0 / (self.sigma ^ 2)
  return self.dv0, deform
end

