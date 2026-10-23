param(
  [string]$ZipPath = (Join-Path $PSScriptRoot '..\..\Data\dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64.zip')
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $ZipPath))
try {
  $entry = $archive.GetEntry('dsv4-flash-dspark-a3-dp4tp4ep16-gbs256-in256o64/result3.json')
  if (-not $entry) { throw 'Archive does not contain result3.json' }
  $input = $entry.Open()
  $output = [IO.File]::Create((Join-Path $PSScriptRoot 'result3.json'))
  try { $input.CopyTo($output) } finally { $output.Dispose(); $input.Dispose() }
} finally { $archive.Dispose() }
node -e "const fs=require('fs'),p=process.argv[1];fs.writeFileSync(p+'/result-data.js','window.RUN_RESULT = '+fs.readFileSync(p+'/result3.json','utf8')+';\\n')" $PSScriptRoot
node (Join-Path $PSScriptRoot 'build-trace-summary.cjs') (Resolve-Path -LiteralPath $ZipPath)
node -e "const fs=require('fs'),p=process.argv[1];fs.writeFileSync(p+'/trace-data.js','window.TRACE_SUMMARY = '+fs.readFileSync(p+'/trace-summary.json','utf8')+';\\n')" $PSScriptRoot
