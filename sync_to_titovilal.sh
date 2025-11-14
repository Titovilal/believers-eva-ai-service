#!/bin/bash

# Script para sincronizar el repo de ExponentiaTeam con titovilal

set -e

EXPONENTIA_DIR="/home/salva/projects/believers-eva-ai-service"
TITOVILAL_DIR="/home/salva/projects/believers-eva-ai-service-titovilal"

echo "🔄 Sincronizando repositorio a titovilal..."

# Ir al directorio de titovilal
cd "$TITOVILAL_DIR"

# Agregar el repo de ExponentiaTeam como remote si no existe
if ! git remote | grep -q "exponentia"; then
    git remote add exponentia "$EXPONENTIA_DIR"
fi

# Fetch desde el repo de ExponentiaTeam
git fetch exponentia master

# Resetear completamente al estado del repo de ExponentiaTeam (overwrite)
git reset --hard exponentia/master

# Push forzado al repo de titovilal
git push -f origin master

echo "✅ Sincronización completada!"
echo "📍 Repo titovilal actualizado: https://github.com/Titovilal/believers-eva-ai-service"
