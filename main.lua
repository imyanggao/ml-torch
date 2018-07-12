
require('loader.init')
require('model.init')
require('learner.init')

option = utility.opt.parse(arg)

trainLoader, validLoader = loader.setup(option)

network, criterion = model.setup(option)

-- network = torch.load('../pretrain/vgg-16.t7'):cuda()
-- print(network:parameters())

trainer = learner.setup(network, criterion, option)

local optionPath = option.log .. '-option-' .. os.date("%Y_%m_%d_%X")
utility.tbl.save(optionPath, option)
print(sys.COLORS.green .. 'Options:')
if option.data.colormap ~= nil then
  local reducedOption = utility.tbl.shallowCopy(option)
  reducedOption.data.colormap.hash =  nil
  print(reducedOption)
else
  print(option)
end

local best, startEpoch, measure = option.optim.bestMeasure, option.optim.iEpoch
for epoch = startEpoch, option.optim.maxEpoch do
  trainer:train(epoch, trainLoader)
  if option.lrFinder == false then
    measure = trainer:test(epoch, validLoader)
    if measure > option.optim.bestMeasure then
      option.optim.bestEpoch = epoch
      option.optim.bestMeasure = measure
      print('>>>>> Best model so far with measure = ' .. measure)
    end
    if epoch % option.checkpointEpoch == 0 or option.optim.bestEpoch == epoch then
      utility.checkpoint.save(network, option)
    end
  end
  collectgarbage()
end

if option.lrFinder == true then
  gnuplot = require('gnuplot')
  gnuplot.raw('set logscale x')
  gnuplot.axis({option.optim.lrMin, option.optim.lrMax,
                trainer.lrFinderRecord[2]:min() - 0.1, trainer.lrFinderRecord[2]:max() + 0.1})
  gnuplot.xlabel('learning rate')
  gnuplot.ylabel('loss')
  gnuplot.title('for model details, check: ' .. optionPath)
  gnuplot.grid(true)
  gnuplot.plot(trainer.lrFinderRecord[1], trainer.lrFinderRecord[2], '-')
end
