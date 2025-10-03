{{/*
Expand the name of the chart.
*/}}
{{- define "portalis-nim.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "portalis-nim.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "portalis-nim.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "portalis-nim.labels" -}}
helm.sh/chart: {{ include "portalis-nim.chart" . }}
{{ include "portalis-nim.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "portalis-nim.selectorLabels" -}}
app.kubernetes.io/name: {{ include "portalis-nim.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: portalis-nim
component: translation-service
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "portalis-nim.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "portalis-nim.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
