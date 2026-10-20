"""Restricted read-only tensor/Replay extraction without loading CUDA DLLs.

Only the five GLOBAL references observed in this local archive are admitted.
Arrays are read-only views of archive bytes; no optimizer/model is restored.
"""
from collections import OrderedDict
import io
import pickle
import zipfile
import numpy as np


class ReplayRecord:
    pass


def tensor(storage, offset, size, stride, requires_grad, hooks, *args):
    value=np.lib.stride_tricks.as_strided(storage[offset:],shape=size,
        strides=tuple(s*storage.dtype.itemsize for s in stride))
    value.setflags(write=False)
    return value


class Reader(pickle.Unpickler):
    def __init__(self, data, archive, prefix):
        super().__init__(io.BytesIO(data))
        self.archive,self.prefix,self.storages=archive,prefix,{}

    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):OrderedDict,
            ('torch._utils','_rebuild_tensor_v2'):tensor,
            ('torch','FloatStorage'):np.dtype('float32'),
            ('torch','ByteStorage'):np.dtype('uint8'),
            ('v4.dqn','ReplayItem'):ReplayRecord}
        if (module,name) not in allowed:raise ValueError(f'Unapproved archive reference {module}.{name}')
        return allowed[(module,name)]

    def persistent_load(self,pid):
        kind,dtype,key,location,count=pid
        assert kind=='storage' and location=='cpu'
        if key not in self.storages:
            value=np.frombuffer(self.archive.read(self.prefix+'data/'+key),dtype=dtype)
            assert len(value)==count
            self.storages[key]=value
        return self.storages[key]


def load(path):
    with zipfile.ZipFile(path) as z:
        key=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        return Reader(z.read(key),z,key.removesuffix('data.pkl')).load()


def forward(state,x):
    x=np.asarray(x,dtype=np.float32)
    assert not np.any(state['input_norm_mean']) and np.all(state['input_norm_std']==1)
    for i in range(3):
        x=x@state[f'layers.{i}.weight'].T+state[f'layers.{i}.bias']
        if i<2:x=np.maximum(x,np.float32(0))
    return x
