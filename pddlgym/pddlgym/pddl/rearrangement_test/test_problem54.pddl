(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	monkey-0 - moveable
	bear-1 - moveable
	pawn-2 - moveable
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
	loc-4-0 - static
	loc-4-1 - static
	loc-4-2 - static
    )

    (:init
    
	(ismonkey monkey-0)
	(isbear bear-1)
	(ispawn pawn-2)
	(isrobot robot)
	(at monkey-0 loc-3-2)
	(at bear-1 loc-3-2)
	(at pawn-2 loc-4-2)
	(at robot loc-4-2)
	(handsfree robot)

    ; action literals
    
	(pick monkey-0)
	(place monkey-0)
	(pick bear-1)
	(place bear-1)
	(pick pawn-2)
	(place pawn-2)
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
	(moveto loc-4-0)
	(moveto loc-4-1)
	(moveto loc-4-2)
    )

    (:goal (and  (at monkey-0 loc-4-1) ))
)
    