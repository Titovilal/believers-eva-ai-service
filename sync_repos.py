#!/usr/bin/env python3
"""
Script para copiar el contenido del repo actual a BELIEVERS.EVA_API,
usar la rama dev-exponentia y hacer force push.
"""

import os
import shutil
import subprocess
import sys


def run_command(cmd, cwd=None, capture_output=True):
    """Ejecuta un comando y retorna el resultado."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=capture_output, text=True
    )
    if result.returncode != 0 and capture_output:
        print(f"Error ejecutando: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    return result


def get_tracked_files(repo_path):
    """Obtiene la lista de archivos trackeados por git."""
    result = run_command("git ls-files", cwd=repo_path)
    if result.returncode == 0:
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    return []


def clean_directory(dest_repo):
    """Limpia el directorio destino excepto .git"""
    print("   Limpiando directorio destino...")
    for item in os.listdir(dest_repo):
        if item == ".git":
            continue
        item_path = os.path.join(dest_repo, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"   Error eliminando {item_path}: {e}")


def copy_files(source_repo, dest_repo, files):
    """Copia los archivos del repo fuente al destino."""
    copied_count = 0
    for file_path in files:
        source_file = os.path.join(source_repo, file_path)
        dest_file = os.path.join(dest_repo, file_path)

        if not os.path.exists(source_file):
            continue

        # Crear directorio destino si no existe
        dest_dir = os.path.dirname(dest_file)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        # Copiar archivo
        shutil.copy2(source_file, dest_file)
        copied_count += 1

    return copied_count


def branch_exists(repo_path, branch_name):
    """Verifica si una rama existe (local o remota)."""
    result = run_command("git branch -a", cwd=repo_path)
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            # Limpiar la línea (quitar *, espacios, remotes/origin/)
            line = line.strip().lstrip("*").strip()
            line = line.replace("remotes/origin/", "")
            if line == branch_name:
                return True
    return False


def main():
    source_repo = "/home/salva/projects/believers-eva-ai-service"
    dest_repo = "/home/salva/projects/BELIEVERS.EVA_API"
    branch_name = "dev-exponentia"

    print("=" * 60)
    print("Sincronizando repos con force overwrite")
    print("=" * 60)

    # Verificar que ambos directorios existen
    if not os.path.exists(source_repo):
        print(f"Error: El directorio fuente no existe: {source_repo}")
        sys.exit(1)

    if not os.path.exists(dest_repo):
        print(f"Error: El directorio destino no existe: {dest_repo}")
        sys.exit(1)

    # Cambiar al directorio destino
    os.chdir(dest_repo)

    # Fetch para obtener info de ramas remotas
    print("\n1. Actualizando referencias remotas...")
    run_command("git fetch origin", cwd=dest_repo, capture_output=False)

    # Verificar si la rama existe
    exists = branch_exists(dest_repo, branch_name)

    if exists:
        print(f"\n2. Rama {branch_name} existe, haciendo checkout...")
        # Intentar checkout, si falla intentar crear desde remota
        result = run_command(
            f"git checkout {branch_name}", cwd=dest_repo, capture_output=False
        )
        if result.returncode != 0:
            print("   Intentando checkout desde remota...")
            result = run_command(
                f"git checkout -b {branch_name} origin/{branch_name}",
                cwd=dest_repo,
                capture_output=False,
            )
            if result.returncode != 0:
                print("Error haciendo checkout de la rama")
                sys.exit(1)
    else:
        print(f"\n2. Creando nueva rama {branch_name}...")
        # Primero asegurarse de estar en una rama base (pro o master)
        run_command(
            "git checkout pro 2>/dev/null || git checkout master",
            cwd=dest_repo,
            capture_output=False,
        )
        result = run_command(
            f"git checkout -b {branch_name}", cwd=dest_repo, capture_output=False
        )
        if result.returncode != 0:
            print("Error creando la rama")
            sys.exit(1)

    # Limpiar directorio
    print("\n3. Limpiando directorio destino...")
    clean_directory(dest_repo)

    # Obtener archivos trackeados
    print("\n4. Obteniendo archivos del repo fuente...")
    files = get_tracked_files(source_repo)
    print(f"   Encontrados {len(files)} archivos trackeados")

    # Copiar archivos
    print("\n5. Copiando archivos...")
    copied = copy_files(source_repo, dest_repo, files)
    print(f"   Copiados {copied} archivos")

    # Agregar todos los cambios (incluyendo eliminaciones)
    print("\n6. Agregando cambios...")
    run_command("git add -A", cwd=dest_repo, capture_output=False)

    # Verificar si hay cambios
    result = run_command("git status --porcelain", cwd=dest_repo)
    if not result.stdout.strip():
        print("   No hay cambios para commitear")
    else:
        # Hacer commit
        print("\n7. Creando commit...")
        commit_msg = "Sync from believers-eva-ai-service (full overwrite)"
        run_command(
            f'git commit -m "{commit_msg}"', cwd=dest_repo, capture_output=False
        )

    # Push force de la rama
    print(f"\n8. Pusheando rama {branch_name} (force)...")
    result = run_command(
        f"git push --force -u origin {branch_name}", cwd=dest_repo, capture_output=False
    )

    if result.returncode == 0:
        print("\n✓ Proceso completado exitosamente!")
        print(f"✓ Rama {branch_name} sincronizada y pusheada con force")
    else:
        print("\n✗ Error pusheando la rama")
        sys.exit(1)


if __name__ == "__main__":
    main()
