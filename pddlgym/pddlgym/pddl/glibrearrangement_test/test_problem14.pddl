(define (problem glibrearrangement) 
    (:domain glibrearrangement)

    (:objects
    
	pawn-0 - moveable
	pawn-1 - moveable
	bear-2 - moveable
	pawn-3 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-0-3 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-1-3 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-2-3 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
	loc-3-3 - static
	loc-4-0 - static
	loc-4-1 - static
	loc-4-2 - static
	loc-4-3 - static
    )

    (:init
    
	(IsPawn pawn-0)
	(IsPawn pawn-1)
	(IsBear bear-2)
	(IsPawn pawn-3)
	(IsRobot robot)
	(At pawn-0 loc-3-0)
	(At pawn-1 loc-2-1)
	(At bear-2 loc-4-3)
	(At pawn-3 loc-3-1)
	(At robot loc-4-1)
	(Handsfree robot)

    ; Action literals
    
	(Pick pawn-0)
	(Place pawn-0)
	(Pick pawn-1)
	(Place pawn-1)
	(Pick bear-2)
	(Place bear-2)
	(Pick pawn-3)
	(Place pawn-3)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-0-3)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-1-3)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
	(MoveTo loc-2-3)
	(MoveTo loc-3-0)
	(MoveTo loc-3-1)
	(MoveTo loc-3-2)
	(MoveTo loc-3-3)
	(MoveTo loc-4-0)
	(MoveTo loc-4-1)
	(MoveTo loc-4-2)
	(MoveTo loc-4-3)
    )

    (:goal (and  (At pawn-3 loc-3-3)  (At pawn-1 loc-0-3)  (At bear-2 loc-3-2) ))
)
    