
(define (problem generatedblocks) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b2 - block
  )
  (:init 
	(clear b0)
	(clear b1)
	(handempty )
	(on b1 b2)
	(ontable b0)
	(ontable b2)
  )
  (:goal (and
	(on b1 b2)
	(on b2 b0)
	(ontable b0)))
)
