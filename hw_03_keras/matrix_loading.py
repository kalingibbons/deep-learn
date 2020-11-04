# %%
from io import StringIO
import numpy as np


# %%
txt = np.random.randn(5, 5).astype(str).tolist()
txt = '\n'.join([','.join(t) for t in txt])
print(txt)


# %%
pd.read_csv(StringIO(txt), header=None).to_numpy()
