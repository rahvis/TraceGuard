// TraceGuard — Azure Confidential VM (AMD SEV-SNP) deployment.
//
// Provisions a genuine Confidential VM with vTPM + Secure Boot + guest attestation,
// then runs the TraceGuard container so its in-guest attestation module can read the
// vTPM, submit evidence to Microsoft Azure Attestation (MAA), and surface a verified
// quote. See deploy/azure/README.md for the one-command wrapper and teardown.

@description('Location for all resources. Must offer the chosen confidential VM size.')
param location string = resourceGroup().location

@description('DNS-safe name prefix; used for the VM, public IP label, and resources.')
@minLength(3)
@maxLength(24)
param namePrefix string = 'traceguard'

@description('Confidential VM size. DCasv5/DCadsv5 = AMD SEV-SNP. Verify availability with `az vm list-skus`.')
param vmSize string = 'Standard_DC4as_v5'

@description('Admin username for SSH.')
param adminUsername string = 'azureuser'

@description('SSH public key (OpenSSH format) for the admin user.')
@secure()
param adminPublicKey string

@description('Canonical Ubuntu CVM image reference (offer:sku). Default: Ubuntu 24.04 CVM.')
param imageOffer string = 'ubuntu-24_04-lts'
param imageSku string = 'cvm'

@description('OS-disk confidential encryption. VMGuestStateOnly encrypts the VMGS; DiskWithVMGuestState also pre-encrypts the OS disk.')
@allowed([
  'VMGuestStateOnly'
  'DiskWithVMGuestState'
])
param osDiskSecurityEncryptionType string = 'VMGuestStateOnly'

@description('Base64-encoded cloud-init that installs Docker + tpm2-tools and runs the container.')
param customData string

@description('CIDR allowed to reach SSH (22). Lock this to your IP; default is open for demos.')
param sshSourceCidr string = '*'

@description('CIDR allowed to reach the app (80/443). Default open so a reviewer can open the URL.')
param appSourceCidr string = '*'

var vmName = '${namePrefix}-cvm'
var nsgName = '${namePrefix}-nsg'
var vnetName = '${namePrefix}-vnet'
var pipName = '${namePrefix}-pip'
var nicName = '${namePrefix}-nic'
var dnsLabel = toLower('${namePrefix}-${uniqueString(resourceGroup().id)}')

resource nsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name: nsgName
  location: location
  properties: {
    securityRules: [
      {
        name: 'allow-ssh'
        properties: {
          priority: 1000
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: sshSourceCidr
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
        }
      }
      {
        name: 'allow-http'
        properties: {
          priority: 1010
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: appSourceCidr
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '80'
        }
      }
      {
        name: 'allow-https'
        properties: {
          priority: 1020
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: appSourceCidr
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '443'
        }
      }
    ]
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: { addressPrefixes: ['10.30.0.0/16'] }
    subnets: [
      {
        name: 'default'
        properties: {
          addressPrefix: '10.30.1.0/24'
          networkSecurityGroup: { id: nsg.id }
        }
      }
    ]
  }
}

resource pip 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: pipName
  location: location
  sku: { name: 'Standard' }
  properties: {
    publicIPAllocationMethod: 'Static'
    dnsSettings: { domainNameLabel: dnsLabel }
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2023-11-01' = {
  name: nicName
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: { id: vnet.properties.subnets[0].id }
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: { id: pip.id }
        }
      }
    ]
  }
}

resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: vmName
  location: location
  properties: {
    hardwareProfile: { vmSize: vmSize }
    osProfile: {
      computerName: vmName
      adminUsername: adminUsername
      customData: customData
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: adminPublicKey
            }
          ]
        }
      }
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: imageOffer
        sku: imageSku
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
          securityProfile: {
            securityEncryptionType: osDiskSecurityEncryptionType
          }
        }
      }
    }
    networkProfile: {
      networkInterfaces: [{ id: nic.id }]
    }
    // The confidential-VM security profile: hardware memory encryption + a vTPM
    // that roots guest attestation, plus Secure Boot for a measured boot chain.
    securityProfile: {
      securityType: 'ConfidentialVM'
      uefiSettings: {
        secureBootEnabled: true
        vTpmEnabled: true
      }
    }
  }
}

// Microsoft guest-attestation extension: attests the boot chain to MAA at boot and
// makes the platform boot-attestation result available. The app additionally runs
// its own in-guest attestation for the live "Verify" action.
resource guestAttestation 'Microsoft.Compute/virtualMachines/extensions@2024-07-01' = {
  parent: vm
  name: 'GuestAttestation'
  location: location
  properties: {
    publisher: 'Microsoft.Azure.Security.LinuxAttestation'
    type: 'GuestAttestation'
    typeHandlerVersion: '1.0'
    autoUpgradeMinorVersion: true
    settings: {
      AttestationConfig: {
        MaaSettings: {
          maaEndpoint: ''
          maaTenantName: 'GuestAttestation'
        }
        disableAlerts: 'false'
      }
    }
  }
}

output publicIp string = pip.properties.ipAddress
output fqdn string = pip.properties.dnsSettings.fqdn
output frontendUrl string = 'http://${pip.properties.dnsSettings.fqdn}'
output vmName string = vmName
output sshCommand string = 'ssh ${adminUsername}@${pip.properties.dnsSettings.fqdn}'
