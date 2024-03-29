


(define (problem fridge-s5-f6)
(:domain fridge-domain)
(:objects 
s0-0 s0-1 s0-2 s0-3 s0-4 s1-0 s1-1 s1-2 s1-3 s1-4 s2-0 s2-1 s2-2 s2-3 s2-4 s3-0 s3-1 s3-2 s3-3 s3-4 s4-0 s4-1 s4-2 s4-3 s4-4 s5-0 s5-1 s5-2 s5-3 s5-4 
- screw
c0-0 c0-1 c1-0 c1-1 c2-0 c2-1 c3-0 c3-1 c4-0 c4-1 c5-0 c5-1 
- compressor
f0 f1 f2 f3 f4 f5 
- fridge)
(:init
(fridge-on f0)
(attached c0-0 f0)
(fridge-on f1)
(attached c1-0 f1)
(fridge-on f2)
(attached c2-0 f2)
(fridge-on f3)
(attached c3-0 f3)
(fridge-on f4)
(attached c4-0 f4)
(fridge-on f5)
(attached c5-0 f5)
(fits s0-0 c0-0)
(fits s0-0 c0-1)
(screwed s0-0)
(fits s0-1 c0-0)
(fits s0-1 c0-1)
(screwed s0-1)
(fits s0-2 c0-0)
(fits s0-2 c0-1)
(screwed s0-2)
(fits s0-3 c0-0)
(fits s0-3 c0-1)
(screwed s0-3)
(fits s0-4 c0-0)
(fits s0-4 c0-1)
(screwed s0-4)
(fits s1-0 c1-0)
(fits s1-0 c1-1)
(screwed s1-0)
(fits s1-1 c1-0)
(fits s1-1 c1-1)
(screwed s1-1)
(fits s1-2 c1-0)
(fits s1-2 c1-1)
(screwed s1-2)
(fits s1-3 c1-0)
(fits s1-3 c1-1)
(screwed s1-3)
(fits s1-4 c1-0)
(fits s1-4 c1-1)
(screwed s1-4)
(fits s2-0 c2-0)
(fits s2-0 c2-1)
(screwed s2-0)
(fits s2-1 c2-0)
(fits s2-1 c2-1)
(screwed s2-1)
(fits s2-2 c2-0)
(fits s2-2 c2-1)
(screwed s2-2)
(fits s2-3 c2-0)
(fits s2-3 c2-1)
(screwed s2-3)
(fits s2-4 c2-0)
(fits s2-4 c2-1)
(screwed s2-4)
(fits s3-0 c3-0)
(fits s3-0 c3-1)
(screwed s3-0)
(fits s3-1 c3-0)
(fits s3-1 c3-1)
(screwed s3-1)
(fits s3-2 c3-0)
(fits s3-2 c3-1)
(screwed s3-2)
(fits s3-3 c3-0)
(fits s3-3 c3-1)
(screwed s3-3)
(fits s3-4 c3-0)
(fits s3-4 c3-1)
(screwed s3-4)
(fits s4-0 c4-0)
(fits s4-0 c4-1)
(screwed s4-0)
(fits s4-1 c4-0)
(fits s4-1 c4-1)
(screwed s4-1)
(fits s4-2 c4-0)
(fits s4-2 c4-1)
(screwed s4-2)
(fits s4-3 c4-0)
(fits s4-3 c4-1)
(screwed s4-3)
(fits s4-4 c4-0)
(fits s4-4 c4-1)
(screwed s4-4)
(fits s5-0 c5-0)
(fits s5-0 c5-1)
(screwed s5-0)
(fits s5-1 c5-0)
(fits s5-1 c5-1)
(screwed s5-1)
(fits s5-2 c5-0)
(fits s5-2 c5-1)
(screwed s5-2)
(fits s5-3 c5-0)
(fits s5-3 c5-1)
(screwed s5-3)
(fits s5-4 c5-0)
(fits s5-4 c5-1)
(screwed s5-4)
)
(:goal
(and
(attached c0-1 f0)
(fridge-on f0)
(attached c1-1 f1)
(fridge-on f1)
(attached c2-1 f2)
(fridge-on f2)
(attached c3-1 f3)
(fridge-on f3)
(attached c4-1 f4)
(fridge-on f4)
(attached c5-1 f5)
(fridge-on f5)
)
)
)


