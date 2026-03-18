param([string]$SpaceArn, [string]$AwsProfile = "")
$ErrorActionPreference = "Stop"
if ($SpaceArn -match "arn:aws:sagemaker:([a-z0-9-]+):") { $Region = $Matches[1] }
$cmd = @("sagemaker", "start-session", "--resource-identifier", $SpaceArn, "--region", $Region)
if ($AwsProfile) { $cmd += @("--profile", $AwsProfile) }
$out = & aws @cmd | ConvertFrom-Json
$json = @{streamUrl=$out.StreamUrl; tokenValue=$out.TokenValue; sessionId=$out.SessionId} | ConvertTo-Json -Compress
& session-manager-plugin $json $Region "StartSession"
