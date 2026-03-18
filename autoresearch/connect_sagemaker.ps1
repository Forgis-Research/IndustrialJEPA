$arn = "arn:aws:sagemaker:eu-central-1:698516610829:space/d-dbpzn6wx78xs/quickstart-gpu-0az0xd"
$region = "eu-central-1"
$profile = "sagemaker"

$out = aws sagemaker start-session --resource-identifier $arn --region $region --profile $profile | ConvertFrom-Json

$jsonObj = @{
    sessionId = $out.SessionId
    streamUrl = $out.StreamUrl
    tokenValue = $out.TokenValue
}

$jsonStr = $jsonObj | ConvertTo-Json -Compress

# Write to temp file
$tempFile = [System.IO.Path]::GetTempFileName()
$jsonStr | Out-File -FilePath $tempFile -Encoding ASCII -NoNewline

# Read back and pass to plugin
$jsonContent = Get-Content $tempFile -Raw
Remove-Item $tempFile

cmd /c "session-manager-plugin '$jsonContent' $region StartSession"
