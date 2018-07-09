
local SemanticSeg, parent = torch.class('learner.SemanticSeg', 'learner.Classification')

function SemanticSeg:__init(model, criterion, option)
  parent.__init(self, model, criterion, option)
end

function SemanticSeg:copyBatch(batchData)
  self.input:resize(batchData.input:size()):copy(batchData.input)
  self.target:resize(batchData.target:size()):copy(batchData.target)
  self.target = self.target:view(-1)
end

function SemanticSeg:example(set, loader)
  local function l2i(img, l, r)
    for i = 1, self.option.example[set].n do
      if self.option.data.colormap ~= nil then
        img[r][i]:copy(self.option.data.colormap:l2c(l[i]))
      else
        if self.option.data.color == true then
          img[r][i]:copy(l[i]:view(1,table.unpack(self.option.example.imageHW))
                           :expand(3,table.unpack(self.option.example.imageHW)))
        else
          -- intensity normalization for prediction, only for display purpose
          img[r][i]:copy(utility.img2d.linearMinMaxNormal()(l[i]))
        end
      end
    end
  end

  if self.option.example[set] ~= nil then
    parent.example(self, set, loader)
    if self.option.example[set].batch == nil then
      self.option.example[set].batch = loader:examples(self.option.example[set].indices)
      
      local g, c, m, h, w, nSave = self.option.example.imageGap, 1, table.unpack(self.option.data.imageSize)
      local hg, wg = h + g, w + g
      if self.option.data.color == true then
        c = 3
        assert(m == 3, 'nChannel is not equal to 3 for color image')
        m = 1
      end
      self.option.example.imageHW = {h, w}

      self.option.example[set].images = torch.ones(c, hg * (m+2) - g, wg * self.option.example[set].n - g):byte() * 255
      self.option.example[set].image = {}
      for i = 1, (m+2) do
        self.option.example[set].image[i] = {}
        for j = 1, self.option.example[set].n do
          self.option.example[set].image[i][j] = self.option.example[set].images[{{}, {(i-1)*hg+1, (i-1)*hg+h}, {(j-1)*wg+1, (j-1)*wg+w}}]
        end
      end
      local fileStr = set
      for i = 1, self.option.example[set].n do
        if self.option.data.color == true then
          self.option.example[set].image[1][i]:copy(self.option.example[set].batch.input[i])
        else
          for j = 1, m do
            -- intensity normalization for gray scale image, only for display purpose
            self.option.example[set].image[j][i]:copy(utility.img2d.linearMinMaxNormal()(self.option.example[set].batch.input[i][j]))
          end
        end
        fileStr = fileStr .. '_' .. self.option.example[set].indices[i]
      end      
      l2i(self.option.example[set].image, self.option.example[set].batch.target, m+1)
      self.option.example[set].predFile = self.option.log .. '-example-pred-' .. fileStr .. '.png'
      self.option.example[set].r = m + 2
      
      if string.find(self.option.example.type, 'all') then
        nSave = math.floor((self.option.optim.maxEpoch - self.option.optim.iEpoch + 1) / self.option.example.epoch)
        self.option.example[set].allImages = torch.ones(c, hg * (m+nSave+1) - g, wg * self.option.example[set].n - g):byte() * 255
        self.option.example[set].allImage = {}
        for i = 1, (m+nSave+1) do
          self.option.example[set].allImage[i] = {}
          for j = 1, self.option.example[set].n do
            self.option.example[set].allImage[i][j] = self.option.example[set].allImages[{{}, {(i-1)*hg+1, (i-1)*hg+h}, {(j-1)*wg+1, (j-1)*wg+w}}]
          end
        end
        for i = 1, self.option.example[set].n do
          if self.option.data.color == true then
            self.option.example[set].allImage[1][i]:copy(self.option.example[set].batch.input[i])
          else
            for j = 1, m do
              -- intensity normalization for gray scale image, only for display purpose
              self.option.example[set].allImage[j][i]:copy(utility.img2d.linearMinMaxNormal()(self.option.example[set].batch.input[i][j]))
            end
          end
        end      
        l2i(self.option.example[set].allImage, self.option.example[set].batch.target, m+1)
        self.option.example[set].allPredFile = self.option.log .. '-example-allpred-' .. fileStr .. '.png'
        self.option.example[set].allR = m + 2
      end
    end
    
    if self.option.optim.iEpoch % self.option.example.epoch == 0 then
      self:copyBatch(self.option.example[set].batch)    
      self.model:forward(self.input)
      local _, prediction = self.model.output:max(2)
      prediction = prediction:view(self.option.example[set].n, table.unpack(self.option.example.imageHW)):byte()
      l2i(self.option.example[set].image, prediction, self.option.example[set].r)
      image.save(self.option.example[set].predFile, self.option.example[set].images)
      if string.find(self.option.example.type, 'all') then
        l2i(self.option.example[set].allImage, prediction, self.option.example[set].allR)
        self.option.example[set].allR = self.option.example[set].allR + 1
        image.save(self.option.example[set].allPredFile, self.option.example[set].allImages)
      end
    end
    
  end
end
