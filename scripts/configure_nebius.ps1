[CmdletBinding()]
param([switch]$ConfigureOnly)

# Run this script in your own PowerShell terminal. Never pass a key as an argument.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$VerbosePreference = 'SilentlyContinue'
$DebugPreference = 'SilentlyContinue'
Set-PSDebug -Off

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$envPath = Join-Path $projectRoot '.env'
$pythonPath = 'D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe'
$smokePath = Join-Path $PSScriptRoot 'nebius_smoke.py'
$secret = $null
$plainKey = $null
$envText = $null
$pointer = [IntPtr]::Zero
$exitCode = 1

function Assert-NoReparsePath([string]$Path) {
    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'Refusing a reparse-point path for the local credential file.'
            }
        }
        $parent = [IO.Directory]::GetParent($current)
        if ($null -eq $parent) { break }
        $current = $parent.FullName
    }
}

try {
    Assert-NoReparsePath $envPath
    $null = & git -C $projectRoot check-ignore --quiet -- .env
    if ($LASTEXITCODE -ne 0) { throw '.env must be ignored before configuration.' }
    $tracked = @(& git -C $projectRoot ls-files -- .env)
    if ($LASTEXITCODE -ne 0 -or $tracked.Count -ne 0) {
        throw 'Refusing to configure a tracked credential file.'
    }
    if (-not $ConfigureOnly) {
        if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf) -or
            -not (Test-Path -LiteralPath $smokePath -PathType Leaf)) {
            throw 'The existing Python environment and smoke-test script are required.'
        }
    }

    Write-Host 'Revoke the previously exposed key and create a new Nebius key first.'
    Write-Host 'Paste only the NEW key into this hidden local prompt; do not send it in chat.'
    $secret = Read-Host -Prompt 'New Nebius API key (hidden)' -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    # Bearer-token characters only: no whitespace, newline or dotenv interpolation.
    if ([string]::IsNullOrEmpty($plainKey) -or $plainKey -cnotmatch '^[A-Za-z0-9._~+/=-]{20,4096}$') {
        throw 'The hidden input was empty or not a single bearer token.'
    }

    $lines = @()
    if (Test-Path -LiteralPath $envPath) {
        if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
            throw '.env must be a regular file.'
        }
        # Keep unrelated settings, but never display or back up existing secrets.
        $lines = @([IO.File]::ReadAllLines($envPath) | Where-Object {
            $_ -notmatch '^\s*(?:export\s+)?NEBIUS_(?:BASE_URL|MODEL|API_KEY)\s*='
        })
    } else {
        $stream = [IO.File]::Open($envPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $stream.Dispose()
    }

    # Restrict permissions BEFORE writing a plaintext key. No secret temp/backup file.
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $systemSid = [Security.Principal.SecurityIdentifier]::new('S-1-5-18')
    $acl = [Security.AccessControl.FileSecurity]::new()
    $acl.SetOwner($identity)
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sid in @($identity, $systemSid)) {
        $rule = [Security.AccessControl.FileSystemAccessRule]::new(
            $sid, [Security.AccessControl.FileSystemRights]::FullControl,
            [Security.AccessControl.AccessControlType]::Allow)
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $envPath -AclObject $acl
    Assert-NoReparsePath $envPath
    $lines += 'NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1'
    $lines += 'NEBIUS_MODEL=nvidia/Nemotron-3_5-Lightning'
    $lines += ('NEBIUS_API_KEY=' + $plainKey)
    $envText = ($lines -join "`n") + "`n"
    [IO.File]::WriteAllText($envPath, $envText, [Text.UTF8Encoding]::new($false))
    $envText = $null
    $lines = $null
    $plainKey = $null
    Write-Host 'Saved local .env with current-user/SYSTEM permissions. The key was not printed.'
    if ($ConfigureOnly) {
        Write-Host 'Configuration saved; no API requests were made.'
        $exitCode = 0
    } else {
        Write-Host 'Starting bounded API smoke tests in the existing CPU Python process.'
        & $pythonPath -B $smokePath
        $exitCode = $LASTEXITCODE
    }
} catch {
    # Do not print exception messages, request headers, local file content or secrets.
    Write-Error 'Nebius setup stopped. Check the ignored .env path, local file permissions, input format, and existing Python/script paths. No key is included in this error.' -ErrorAction Continue
    $exitCode = 1
} finally {
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    if ($null -ne $secret) { $secret.Dispose() }
    $plainKey = $null
    $envText = $null
    $lines = $null
}
exit $exitCode
