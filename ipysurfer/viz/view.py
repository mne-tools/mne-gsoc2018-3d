from collections import namedtuple


View = namedtuple('View', 'elev azim')

views_dict = {'lateral': View(elev=5, azim=0),
              'medial': View(elev=5, azim=180),
              'rostral': View(elev=5, azim=90),
              'caudal': View(elev=5, azim=-90),
              'dorsal': View(elev=90, azim=0),
              'ventral': View(elev=-90, azim=0),
              'frontal': View(elev=5, azim=110),
              'parietal': View(elev=5, azim=-110)}
