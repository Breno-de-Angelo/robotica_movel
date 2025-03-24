from setuptools import find_packages, setup

package_name = 'controlador_quadrimotor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='breno',
    maintainer_email='brenodeangelo@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seguimento_trajetoria = controlador_quadrimotor.seguimento_trajetoria:main',
            'seguimento_caminho = controlador_quadrimotor.seguimento_caminho:main',
            'simulador = controlador_quadrimotor.simulador:main',
        ],
    },
)
