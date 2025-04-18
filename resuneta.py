# Arquitectura de red ResUnet
import torch
import torch.nn as nn
import torch.nn.functional as F

class Initialization(nn.Module):
    """
    Capa inicial para transformar las 4 bandas de entrada en el número inicial de filtros.

    Args:
        output_num_filters (int): Número de filtros de salida.

    Attributes:
        convolution (nn.Conv2d): Convolución 1x1 para ajustar el número de canales.
    """

    def __init__(self, output_num_filters):
        super().__init__()
        self.convolution = nn.Conv2d(4, output_num_filters, (1, 1), bias=True)

    def forward(self, input):
        """
        Procesa la entrada a través de la convolución inicial.

        Args:
            input (torch.Tensor): Tensor de entrada con forma (batch_size, 4, H, W).

        Returns:
            torch.Tensor: Salida con forma (batch_size, output_num_filters, H, W).
        """
        return self.convolution(input)

class ResNetBlock(nn.Module):
    """
    Bloque residual con dos convoluciones, normalización por instancia y dropout.

    Args:
        num_filters (int): Número de filtros en las convoluciones.
        dilation (int): Tasa de dilatación para las convoluciones.

    Attributes:
        instance_norm1/2 (nn.InstanceNorm2d): Normalización por instancia.
        conv1/2 (nn.Conv2d): Capas de convolución con dilatación.
        dropout (nn.Dropout2d): Dropout para regularización.
    """
    def __init__(self, num_filters, dilation):
        super().__init__()
        self.instance_norm1 = nn.InstanceNorm2d(num_filters, affine=True)
        self.conv1 = nn.Conv2d(num_filters, num_filters, (3, 3), dilation=dilation, padding=dilation, bias=False)
        self.instance_norm2 = nn.InstanceNorm2d(num_filters, affine=True)
        self.dropout = nn.Dropout2d(p=0.25)  # Aplicar dropout antes de la última convolución
        self.conv2 = nn.Conv2d(num_filters, num_filters, (3, 3), dilation=dilation, padding=dilation, bias=False)

    def forward(self, x):
        """
        Procesa la entrada a través del bloque residual.

        Args:
            x (torch.Tensor): Entrada del bloque.

        Returns:
            torch.Tensor: Salida del bloque con conexión residual añadida.
        """
        identity = x
        out = self.instance_norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.instance_norm2(out)
        out = F.relu(out)
        out = self.dropout(out) # Aplicar dropout antes de la última convolución
        out = self.conv2(out)
        out += identity # Conexión residua
        return out

class Upscaling(nn.Module):
    """
    Capa de upscaling usando convolución transpuesta.

    Args:
        input_num_filters (int): Número de filtros de entrada.
        output_num_filters (int): Número de filtros de salida.
    """
    def __init__(self, input_num_filters, output_num_filters):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(input_num_filters, output_num_filters, (2, 2), stride=(2, 2), bias=False)

    def forward(self, input):
        return self.conv_transpose(input)

class Downscaling(nn.Module):
    """
    Capa de downscaling usando convolución con stride.

    Args:
        input_num_filters (int): Número de filtros de entrada.
        output_num_filters (int): Número de filtros de salida.
    """
    def __init__(self, input_num_filters, output_num_filters):
        super().__init__()
        self.conv = nn.Conv2d(input_num_filters, output_num_filters, (2, 2), stride=(2, 2), bias=False)

    def forward(self, input):
        return self.conv(input)

class Combining(nn.Module):
    """
    Combina características de diferentes niveles usando concatenación y convolución.

    Args:
        first_num_filters (int): Número de filtros del primer tensor.
        second_num_filters (int): Número de filtros del segundo tensor.
        output_num_filters (int): Número de filtros de salida.
    """
    def __init__(self, first_num_filters, second_num_filters, output_num_filters):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(
            first_num_filters + second_num_filters, affine=True
        )
        self.conv = nn.Conv2d(
            first_num_filters + second_num_filters, output_num_filters, (3, 3), padding=1, bias=False
        )  # Cambié de (1,1) a (3,3) para mejorar la fusión

    def forward(self, first, second):
        """
        Concatena y procesa dos tensores.

        Args:
            first (torch.Tensor): Primer tensor de entrada.
            second (torch.Tensor): Segundo tensor de entrada.

        Returns:
            torch.Tensor: Salida combinada.
        """
        output = torch.cat([first, second], dim=1)
        output = self.instance_norm(output)
        output = self.conv(output)
        return output

class ResUNetA(nn.Module):
    """
    Arquitectura completa ResUNet para segmentación.

    Args:
        num_filters (list): Lista de números de filtros por etapa.
        dilations (list): Lista de tasas de dilatación por etapa.

    Attributes:
        initialization (Initialization): Capa inicial.
        down_res_blocks (nn.ModuleList): Bloques residuales de downscaling.
        downscalings (nn.ModuleList): Capas de downscaling.
        up_res_blocks (nn.ModuleList): Bloques residuales de upscaling.
        upscalings (nn.ModuleList): Capas de upscaling.
        combinings (nn.ModuleList): Capas de combinación.
        final_dropout (nn.Dropout2d): Dropout final.
        final_convolution (nn.Conv2d): Convolución final para salida.
    """
    def __init__(self, num_filters=[16, 32, 64, 128, 256], dilations=[1, 1, 1, 2, 2]):
        super().__init__()
        self._num_filters = num_filters
        self._dilations = dilations
        self._num_stages = len(self._num_filters)

        self.initialization = Initialization(self._num_filters[0])
        self.down_res_blocks = nn.ModuleList([ResNetBlock(num_filters[i], dilations[i]) for i in range(self._num_stages)])
        self.downscalings = nn.ModuleList([Downscaling(self._num_filters[i], self._num_filters[i + 1]) for i in range(self._num_stages - 1)])
        self.up_res_blocks = nn.ModuleList([ResNetBlock(num_filters[i], dilations[i]) for i in range(self._num_stages - 1)])
        self.upscalings = nn.ModuleList([Upscaling(self._num_filters[i + 1], self._num_filters[i]) for i in range(self._num_stages - 1)])
        self.combinings = nn.ModuleList([Combining(self._num_filters[i], self._num_filters[i], self._num_filters[i]) for i in range(self._num_stages - 1)])

        self.final_dropout = nn.Dropout2d(p=0.25)  # Dropout final para regularización
        self.final_convolution = nn.Conv2d(self._num_filters[0], 1, (1, 1), bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Inicializa los pesos de las capas convolucionales y normalizaciones.

        Usa Kaiming normal para convoluciones y constantes para biases y normalizaciones.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        """
        Procesa la entrada a través de la red ResUNet.

        Args:
            input (torch.Tensor): Tensor de entrada con forma (batch_size, 4, H, W).

        Returns:
            torch.Tensor: Salida con forma (batch_size, 1, H, W) para segmentación binaria.
        """
        down_outputs = []
        output = self.initialization(input)
        for i in range(self._num_stages - 1):
            output = self.down_res_blocks[i](output)
            down_outputs.append(output)
            output = self.downscalings[i](output)
        output = self.down_res_blocks[self._num_stages - 1](output)

        for i in reversed(range(self._num_stages - 1)):
            output = self.upscalings[i](output)
            output = self.combinings[i](output, down_outputs[i])
            output = self.up_res_blocks[i](output)

        output = self.final_dropout(output)
        output = self.final_convolution(output)
        return output

    def predict(self, input, threshold=0.5):
        """
        Realiza predicciones con un umbral dado.

        Args:
            input (torch.Tensor): Imágenes de entrada.
            threshold (float): Umbral para binarizar la salida.

        Returns:
            torch.Tensor: Máscaras binarias predichas.
        """
        self.eval()
        with torch.no_grad():
            output = self(input)
            output = torch.sigmoid(output)
            mask = (output > threshold).float()
        return mask

    def _predict(self, input, threshold):
        """
        Método interno para predicciones con umbral dinámico (usado después del entrenamiento).

        Args:
            input (torch.Tensor): Imágenes de entrada.
            threshold (float): Umbral optimizado.

        Returns:
            torch.Tensor: Máscaras binarias.
        """
        return self.predict(input, threshold)