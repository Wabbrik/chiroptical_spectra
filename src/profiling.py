import cProfile, io, pstats
from pstats import SortKey
import main

with cProfile.Profile() as pr:
    main.main()
    s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s)
ps.sort_stats(SortKey.CUMULATIVE).print_stats(50)
ps.sort_stats(SortKey.TIME).print_stats(50)
print(s.getvalue())
