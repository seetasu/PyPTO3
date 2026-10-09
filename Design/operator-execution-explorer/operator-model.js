// Source excerpt verified against the upstream pypto-lib file. sourceRefs are excerpt-local.
export const sourceLines = [
  'import pypto.language as pl', '',
  'M = 256', 'N = 256', 'K = 256', 'M_TILE = 64', 'N_TILE = 64', '',
  '@pl.jit', 'def matmul(',
  '    a: pl.Tensor[[M, K], pl.FP32],',
  '    b: pl.Tensor[[K, N], pl.FP32],',
  '    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],',
  '):',
  '    for mb in pl.parallel(0, M, M_TILE):',
  '        for nb in pl.parallel(0, N, N_TILE):',
  '            with pl.at(level=pl.Level.CORE_GROUP, name_hint="matmul_tile"):',
  '                tile_a = a[mb : mb + M_TILE, :]',
  '                tile_b = b[:, nb : nb + N_TILE]',
  '                c[mb : mb + M_TILE, nb : nb + N_TILE] = pl.matmul(tile_a, tile_b)',
  '    return c'
];
const region = (tensor, rows, cols) => ({ tensor, rows, cols, shape: [rows[1]-rows[0], cols[1]-cols[0]], evidenceLevel: 'DERIVED' });
const workUnits = Array.from({length: 16}, (_, i) => {
  const m = Math.floor(i / 4), n = i % 4;
  return {id: `M${m}N${n}`, indices: {m,n}, output: region('C',[m*64,m*64+64],[n*64,n*64+64]), inputs: {A:region('A',[m*64,m*64+64],[0,256]), B:region('B',[0,256],[n*64,n*64+64])}, evidenceLevel:'DERIVED'};
});
export const operatorModel = {
  operator:{name:'MatMul',language:'PyPTO 3.0',evidenceLevel:'SOURCE'},
  source:{path:'examples/beginner/matmul.py',url:'https://github.com/hw-native-sys/pypto-lib/blob/main/examples/beginner/matmul.py',provenance:'Excerpt verified against pypto-lib/examples/beginner/matmul.py on GitHub; line numbers below are excerpt-local',lines:sourceLines,evidenceLevel:'SOURCE'},
  dimensions:{M:256,N:256,K:256,M_TILE:64,N_TILE:64,evidenceLevel:'SOURCE'},
  tensors:Object.fromEntries(['A','B','C'].map((id)=>[id,{identity:id,role:id==='C'?'output':'input',logicalShape:[256,256],coordinates:null,slice:null,dtype:'FP32',layout:null,memoryLocation:null,transformation:null,sourceRefs:id==='A'?[11,18]:id==='B'?[12,19]:[13,20],evidenceLevel:'SOURCE',compilerEvidence:null,runtimeEvidence:null}])),
  parallelAxes:[{axis:'M',count:4,sourceRefs:[15],evidenceLevel:'SOURCE'},{axis:'N',count:4,sourceRefs:[16],evidenceLevel:'SOURCE'}],
  workUnits,selectedWorkUnit:'M1N3',
  tiling:{output:[64,64],A:[64,256],B:[256,64],K:{sourceLevel:256,explicitTile:null,compilerManaged:true},evidenceLevel:'DERIVED'},
  steps:[
    {id:'output',label:'Select Output Tile',sourceRefs:[20],evidenceLevel:'SOURCE'},
    {id:'enter',label:'Enter CORE_GROUP',sourceRefs:[17],evidenceLevel:'SOURCE'},
    {id:'sliceA',label:'Slice A',sourceRefs:[18],evidenceLevel:'SOURCE'},
    {id:'sliceB',label:'Slice B',sourceRefs:[19],evidenceLevel:'SOURCE'},
    {id:'matmul',label:'MatMul',sourceRefs:[20],evidenceLevel:'SOURCE'},
    {id:'write',label:'Write C',sourceRefs:[20],evidenceLevel:'SOURCE'}
  ],
  transformations:{output:'Select output region',sliceA:'Slice rows',sliceB:'Slice columns',matmul:'Matrix multiplication',write:'Write result'},
  performanceModel:{computeOps:33554432,perTileComputeOps:2097152,tensorBytes:null,arithmeticIntensity:null,parallelWork:16,resourceEstimate:null,hardwarePeak:null,estimatedUtilization:null,evidenceLevel:'DERIVED'},
  sourceRefs:{parallel:[15,16],enter:[17],sliceA:[18],sliceB:[19],matmul:[20],write:[20],overview:[1,3,4,5,6,7,11,12,13,15,16,17,18,19,20]},
  compilerEvidence:null,runtimeEvidence:null
};
