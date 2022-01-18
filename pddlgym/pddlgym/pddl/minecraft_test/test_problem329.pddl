(define (problem minecraft) 
    (:domain minecraft)

    (:objects
    
	grass-0 - moveable
	log-1 - moveable
	grass-2 - moveable
	log-3 - moveable
	grass-4 - moveable
	new-0 - moveable
	new-1 - moveable
	new-2 - moveable
	agent - agent
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
    )

    (:init
    
	(hypothetical new-0)
	(hypothetical new-1)
	(hypothetical new-2)
	(isgrass grass-0)
	(islog log-1)
	(isgrass grass-2)
	(islog log-3)
	(isgrass grass-4)
	(at grass-0 loc-2-1)
	(at log-1 loc-1-2)
	(at grass-2 loc-2-2)
	(at log-3 loc-2-3)
	(at grass-4 loc-1-3)
	(agentat loc-2-3)
	(handsfree agent)

    ; action literals
    
	(recall grass-0)
	(craftplank grass-0 log-1)
	(craftplank grass-0 grass-2)
	(craftplank grass-0 log-3)
	(craftplank grass-0 grass-4)
	(craftplank grass-0 new-0)
	(craftplank grass-0 new-1)
	(craftplank grass-0 new-2)
	(equip grass-0)
	(pick grass-0)
	(recall log-1)
	(craftplank log-1 grass-0)
	(craftplank log-1 grass-2)
	(craftplank log-1 log-3)
	(craftplank log-1 grass-4)
	(craftplank log-1 new-0)
	(craftplank log-1 new-1)
	(craftplank log-1 new-2)
	(equip log-1)
	(pick log-1)
	(recall grass-2)
	(craftplank grass-2 grass-0)
	(craftplank grass-2 log-1)
	(craftplank grass-2 log-3)
	(craftplank grass-2 grass-4)
	(craftplank grass-2 new-0)
	(craftplank grass-2 new-1)
	(craftplank grass-2 new-2)
	(equip grass-2)
	(pick grass-2)
	(recall log-3)
	(craftplank log-3 grass-0)
	(craftplank log-3 log-1)
	(craftplank log-3 grass-2)
	(craftplank log-3 grass-4)
	(craftplank log-3 new-0)
	(craftplank log-3 new-1)
	(craftplank log-3 new-2)
	(equip log-3)
	(pick log-3)
	(recall grass-4)
	(craftplank grass-4 grass-0)
	(craftplank grass-4 log-1)
	(craftplank grass-4 grass-2)
	(craftplank grass-4 log-3)
	(craftplank grass-4 new-0)
	(craftplank grass-4 new-1)
	(craftplank grass-4 new-2)
	(equip grass-4)
	(pick grass-4)
	(recall new-0)
	(craftplank new-0 grass-0)
	(craftplank new-0 log-1)
	(craftplank new-0 grass-2)
	(craftplank new-0 log-3)
	(craftplank new-0 grass-4)
	(craftplank new-0 new-1)
	(craftplank new-0 new-2)
	(equip new-0)
	(pick new-0)
	(recall new-1)
	(craftplank new-1 grass-0)
	(craftplank new-1 log-1)
	(craftplank new-1 grass-2)
	(craftplank new-1 log-3)
	(craftplank new-1 grass-4)
	(craftplank new-1 new-0)
	(craftplank new-1 new-2)
	(equip new-1)
	(pick new-1)
	(recall new-2)
	(craftplank new-2 grass-0)
	(craftplank new-2 log-1)
	(craftplank new-2 grass-2)
	(craftplank new-2 log-3)
	(craftplank new-2 grass-4)
	(craftplank new-2 new-0)
	(craftplank new-2 new-1)
	(equip new-2)
	(pick new-2)
	(move loc-0-0)
	(move loc-0-1)
	(move loc-0-2)
	(move loc-0-3)
	(move loc-1-0)
	(move loc-1-1)
	(move loc-1-2)
	(move loc-1-3)
	(move loc-2-0)
	(move loc-2-1)
	(move loc-2-2)
	(move loc-2-3)
    )

    (:goal (and  (inventory grass-4)  (equipped new-1 agent) ))
)
    