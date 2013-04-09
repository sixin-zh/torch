local ClassNLLCriterion, parent = torch.class('nn.ClassNLLCriterion', 'nn.Criterion')

function ClassNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
   self.output = 0
end

function ClassNLLCriterion:updateOutput(input, target)
   input.nn.ClassNLLCriterion_updateOutput(self, input, target)
   return self.output
end

function ClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   input.nn.ClassNLLCriterion_updateGradInput(self, input, target)
   return self.gradInput
end
