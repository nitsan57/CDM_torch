(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	bear-0 - moveable
	pawn-1 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
    )

    (:init
    
	(isbear bear-0)
	(ispawn pawn-1)
	(isrobot robot)
	(at bear-0 loc-1-2)
	(at pawn-1 loc-0-2)
	(at robot loc-3-0)
	(handsfree robot)

    ; action literals
    
	(pick bear-0)
	(place bear-0)
	(pick pawn-1)
	(place pawn-1)
	(moveto loc-0-0)
	(moveto loc-0-1)
	(moveto loc-0-2)
	(moveto loc-1-0)
	(moveto loc-1-1)
	(moveto loc-1-2)
	(moveto loc-2-0)
	(moveto loc-2-1)
	(moveto loc-2-2)
	(moveto loc-3-0)
	(moveto loc-3-1)
	(moveto loc-3-2)
    )

    (:goal (and  (holding pawn-1)  (at bear-0 loc-2-0) ))
)
    