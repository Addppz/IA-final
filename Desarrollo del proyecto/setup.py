#!/usr/bin/env python3
"""
Script de instalación rápida para la aplicación de Predicción de Depósitos Bancarios
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True

def create_venv():
    """Crea un entorno virtual"""
    if os.path.exists("venv"):
        print("ℹ️ Entorno virtual ya existe")
        return True
    
    return run_command("python -m venv venv", "Creando entorno virtual")

def activate_venv():
    """Activa el entorno virtual"""
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
    
    print(f"ℹ️ Para activar el entorno virtual, ejecuta: {activate_script}")
    return True

def install_dependencies():
    """Instala las dependencias"""
    return run_command("pip install -r requirements.txt", "Instalando dependencias")

def main():
    """Función principal"""
    print("🚀 Configuración de la Aplicación de Predicción de Depósitos Bancarios")
    print("=" * 70)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Crear entorno virtual
    if not create_venv():
        sys.exit(1)
    
    # Activar entorno virtual
    activate_venv()
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error al instalar dependencias")
        sys.exit(1)
    
    print("\n🎉 ¡Configuración completada exitosamente!")
    print("\n📋 Próximos pasos:")
    print("1. Activa el entorno virtual:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("2. Ejecuta la aplicación:")
    print("   streamlit run app.py")
    
    print("\n🌐 La aplicación se abrirá en: http://localhost:8501")
    print("\n📚 Para más información, consulta el README.md")

if __name__ == "__main__":
    main() 