
local metric = {}

function metric.acc(cm)
  cm = cm:double()
  return cm:trace() / cm:sum()
end

function metric.meanAcc(cm)
  cm = cm:double()
  return cm:diag():cdiv(cm:sum(2)):mean()
end

function metric.meanIntersectOverUnion(cm)
  cm = cm:double()
  return cm:diag():cdiv(cm:sum(1) + cm:sum(2) - cm:diag()):mean()
end

function metric.freqWeightedIntersectOverUnion(cm)
  cm = cm:double()
  return cm:diag():cmul(cm:sum(2)):cdiv(cm:sum(1) + cm:sum(2) - cm:diag()):sum() / cm:sum()
end

utility.metric = metric
