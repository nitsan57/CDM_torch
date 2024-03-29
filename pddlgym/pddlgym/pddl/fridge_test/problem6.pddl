


(define (problem fridge-s10-f5)
(:domain fridge-domain)
(:objects 
s0-0 s0-1 s0-2 s0-3 s0-4 s0-5 s0-6 s0-7 s0-8 s0-9 s1-0 s1-1 s1-2 s1-3 s1-4 s1-5 s1-6 s1-7 s1-8 s1-9 s2-0 s2-1 s2-2 s2-3 s2-4 s2-5 s2-6 s2-7 s2-8 s2-9 s3-0 s3-1 s3-2 s3-3 s3-4 s3-5 s3-6 s3-7 s3-8 s3-9 s4-0 s4-1 s4-2 s4-3 s4-4 s4-5 s4-6 s4-7 s4-8 s4-9 
- screw
c0-0 c0-1 c1-0 c1-1 c2-0 c2-1 c3-0 c3-1 c4-0 c4-1 
- compressor
f0 f1 f2 f3 f4 
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
(fits s0-5 c0-0)
(fits s0-5 c0-1)
(screwed s0-5)
(fits s0-6 c0-0)
(fits s0-6 c0-1)
(screwed s0-6)
(fits s0-7 c0-0)
(fits s0-7 c0-1)
(screwed s0-7)
(fits s0-8 c0-0)
(fits s0-8 c0-1)
(screwed s0-8)
(fits s0-9 c0-0)
(fits s0-9 c0-1)
(screwed s0-9)
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
(fits s1-5 c1-0)
(fits s1-5 c1-1)
(screwed s1-5)
(fits s1-6 c1-0)
(fits s1-6 c1-1)
(screwed s1-6)
(fits s1-7 c1-0)
(fits s1-7 c1-1)
(screwed s1-7)
(fits s1-8 c1-0)
(fits s1-8 c1-1)
(screwed s1-8)
(fits s1-9 c1-0)
(fits s1-9 c1-1)
(screwed s1-9)
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
(fits s2-5 c2-0)
(fits s2-5 c2-1)
(screwed s2-5)
(fits s2-6 c2-0)
(fits s2-6 c2-1)
(screwed s2-6)
(fits s2-7 c2-0)
(fits s2-7 c2-1)
(screwed s2-7)
(fits s2-8 c2-0)
(fits s2-8 c2-1)
(screwed s2-8)
(fits s2-9 c2-0)
(fits s2-9 c2-1)
(screwed s2-9)
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
(fits s3-5 c3-0)
(fits s3-5 c3-1)
(screwed s3-5)
(fits s3-6 c3-0)
(fits s3-6 c3-1)
(screwed s3-6)
(fits s3-7 c3-0)
(fits s3-7 c3-1)
(screwed s3-7)
(fits s3-8 c3-0)
(fits s3-8 c3-1)
(screwed s3-8)
(fits s3-9 c3-0)
(fits s3-9 c3-1)
(screwed s3-9)
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
(fits s4-5 c4-0)
(fits s4-5 c4-1)
(screwed s4-5)
(fits s4-6 c4-0)
(fits s4-6 c4-1)
(screwed s4-6)
(fits s4-7 c4-0)
(fits s4-7 c4-1)
(screwed s4-7)
(fits s4-8 c4-0)
(fits s4-8 c4-1)
(screwed s4-8)
(fits s4-9 c4-0)
(fits s4-9 c4-1)
(screwed s4-9)
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
)
)
)


