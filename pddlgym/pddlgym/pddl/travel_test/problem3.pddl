
(define (problem travel) (:domain travel)
  (:objects
        az - state
	ca - state
	car-0 - car
	car-1 - car
	ky - state
	nj - state
	nm - state
	og - state
	pe - state
	plane-0 - plane
	plane-1 - plane
	tn - state
	tx - state
	wv - state
  )
  (:goal (and
	(visited tx)
	(visited tn)
	(visited nm)))
  (:init 
	(drive az az car-0)
	(drive az az car-1)
	(drive az ca car-0)
	(drive az ca car-1)
	(drive az ky car-0)
	(drive az ky car-1)
	(drive az nj car-0)
	(drive az nj car-1)
	(drive az nm car-0)
	(drive az nm car-1)
	(drive az og car-0)
	(drive az og car-1)
	(drive az pe car-0)
	(drive az pe car-1)
	(drive az tn car-0)
	(drive az tn car-1)
	(drive az tx car-0)
	(drive az tx car-1)
	(drive az wv car-0)
	(drive az wv car-1)
	(drive ca az car-0)
	(drive ca az car-1)
	(drive ca ca car-0)
	(drive ca ca car-1)
	(drive ca ky car-0)
	(drive ca ky car-1)
	(drive ca nj car-0)
	(drive ca nj car-1)
	(drive ca nm car-0)
	(drive ca nm car-1)
	(drive ca og car-0)
	(drive ca og car-1)
	(drive ca pe car-0)
	(drive ca pe car-1)
	(drive ca tn car-0)
	(drive ca tn car-1)
	(drive ca tx car-0)
	(drive ca tx car-1)
	(drive ca wv car-0)
	(drive ca wv car-1)
	(drive ky az car-0)
	(drive ky az car-1)
	(drive ky ca car-0)
	(drive ky ca car-1)
	(drive ky ky car-0)
	(drive ky ky car-1)
	(drive ky nj car-0)
	(drive ky nj car-1)
	(drive ky nm car-0)
	(drive ky nm car-1)
	(drive ky og car-0)
	(drive ky og car-1)
	(drive ky pe car-0)
	(drive ky pe car-1)
	(drive ky tn car-0)
	(drive ky tn car-1)
	(drive ky tx car-0)
	(drive ky tx car-1)
	(drive ky wv car-0)
	(drive ky wv car-1)
	(drive nj az car-0)
	(drive nj az car-1)
	(drive nj ca car-0)
	(drive nj ca car-1)
	(drive nj ky car-0)
	(drive nj ky car-1)
	(drive nj nj car-0)
	(drive nj nj car-1)
	(drive nj nm car-0)
	(drive nj nm car-1)
	(drive nj og car-0)
	(drive nj og car-1)
	(drive nj pe car-0)
	(drive nj pe car-1)
	(drive nj tn car-0)
	(drive nj tn car-1)
	(drive nj tx car-0)
	(drive nj tx car-1)
	(drive nj wv car-0)
	(drive nj wv car-1)
	(drive nm az car-0)
	(drive nm az car-1)
	(drive nm ca car-0)
	(drive nm ca car-1)
	(drive nm ky car-0)
	(drive nm ky car-1)
	(drive nm nj car-0)
	(drive nm nj car-1)
	(drive nm nm car-0)
	(drive nm nm car-1)
	(drive nm og car-0)
	(drive nm og car-1)
	(drive nm pe car-0)
	(drive nm pe car-1)
	(drive nm tn car-0)
	(drive nm tn car-1)
	(drive nm tx car-0)
	(drive nm tx car-1)
	(drive nm wv car-0)
	(drive nm wv car-1)
	(drive og az car-0)
	(drive og az car-1)
	(drive og ca car-0)
	(drive og ca car-1)
	(drive og ky car-0)
	(drive og ky car-1)
	(drive og nj car-0)
	(drive og nj car-1)
	(drive og nm car-0)
	(drive og nm car-1)
	(drive og og car-0)
	(drive og og car-1)
	(drive og pe car-0)
	(drive og pe car-1)
	(drive og tn car-0)
	(drive og tn car-1)
	(drive og tx car-0)
	(drive og tx car-1)
	(drive og wv car-0)
	(drive og wv car-1)
	(drive pe az car-0)
	(drive pe az car-1)
	(drive pe ca car-0)
	(drive pe ca car-1)
	(drive pe ky car-0)
	(drive pe ky car-1)
	(drive pe nj car-0)
	(drive pe nj car-1)
	(drive pe nm car-0)
	(drive pe nm car-1)
	(drive pe og car-0)
	(drive pe og car-1)
	(drive pe pe car-0)
	(drive pe pe car-1)
	(drive pe tn car-0)
	(drive pe tn car-1)
	(drive pe tx car-0)
	(drive pe tx car-1)
	(drive pe wv car-0)
	(drive pe wv car-1)
	(drive tn az car-0)
	(drive tn az car-1)
	(drive tn ca car-0)
	(drive tn ca car-1)
	(drive tn ky car-0)
	(drive tn ky car-1)
	(drive tn nj car-0)
	(drive tn nj car-1)
	(drive tn nm car-0)
	(drive tn nm car-1)
	(drive tn og car-0)
	(drive tn og car-1)
	(drive tn pe car-0)
	(drive tn pe car-1)
	(drive tn tn car-0)
	(drive tn tn car-1)
	(drive tn tx car-0)
	(drive tn tx car-1)
	(drive tn wv car-0)
	(drive tn wv car-1)
	(drive tx az car-0)
	(drive tx az car-1)
	(drive tx ca car-0)
	(drive tx ca car-1)
	(drive tx ky car-0)
	(drive tx ky car-1)
	(drive tx nj car-0)
	(drive tx nj car-1)
	(drive tx nm car-0)
	(drive tx nm car-1)
	(drive tx og car-0)
	(drive tx og car-1)
	(drive tx pe car-0)
	(drive tx pe car-1)
	(drive tx tn car-0)
	(drive tx tn car-1)
	(drive tx tx car-0)
	(drive tx tx car-1)
	(drive tx wv car-0)
	(drive tx wv car-1)
	(drive wv az car-0)
	(drive wv az car-1)
	(drive wv ca car-0)
	(drive wv ca car-1)
	(drive wv ky car-0)
	(drive wv ky car-1)
	(drive wv nj car-0)
	(drive wv nj car-1)
	(drive wv nm car-0)
	(drive wv nm car-1)
	(drive wv og car-0)
	(drive wv og car-1)
	(drive wv pe car-0)
	(drive wv pe car-1)
	(drive wv tn car-0)
	(drive wv tn car-1)
	(drive wv tx car-0)
	(drive wv tx car-1)
	(drive wv wv car-0)
	(drive wv wv car-1)
	(fly az plane-0)
	(fly az plane-1)
	(fly ca plane-0)
	(fly ca plane-1)
	(fly ky plane-0)
	(fly ky plane-1)
	(fly nj plane-0)
	(fly nj plane-1)
	(fly nm plane-0)
	(fly nm plane-1)
	(fly og plane-0)
	(fly og plane-1)
	(fly pe plane-0)
	(fly pe plane-1)
	(fly tn plane-0)
	(fly tn plane-1)
	(fly tx plane-0)
	(fly tx plane-1)
	(fly wv plane-0)
	(fly wv plane-1)
	(walk az)
	(walk ca)
	(walk ky)
	(walk nj)
	(walk nm)
	(walk og)
	(walk pe)
	(walk tn)
	(walk tx)
	(walk wv)
	(adjacent az ca)
	(adjacent az nm)
	(adjacent ca az)
	(adjacent ca og)
	(adjacent ky tn)
	(adjacent ky wv)
	(adjacent nj pe)
	(adjacent nm az)
	(adjacent nm tx)
	(adjacent og ca)
	(adjacent pe nj)
	(adjacent pe wv)
	(adjacent tn ky)
	(adjacent tx nm)
	(adjacent wv ky)
	(adjacent wv pe)
	(at nj)
	(caravailable car-0)
	(caravailable car-1)
	(isblueplane plane-1)
	(isbluestate ca)
	(isbluestate nm)
	(isbluestate tx)
	(isredplane plane-0)
	(isredstate az)
	(isredstate ky)
	(isredstate og)
	(planeavailable plane-0)
	(planeavailable plane-1)
))
        