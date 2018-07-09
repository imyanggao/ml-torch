
-- require('torch')

require('optim')

learner = {}

require('learner.Supervised')
require('learner.Classification')
require('learner.SemanticSeg')
require('learner.Sequence')
require('learner.SeqLSTM')
require('learner.SeqLSTMCls')

function learner.setup(network, criterion, option)
  local trainer
  if option.model.network == 'VGG' then
    trainer = learner.Classification
  elseif option.model.network == 'FCNVGG' or option.model.network == 'UNet' then
    trainer = learner.SemanticSeg
  elseif option.model.network == 'FCSLSTMVGG' then
    trainer = learner.SeqLSTMCls
  end
  return trainer(network, criterion, option)
end

return learner
