import cv2

class Visualize_Simulation:
    """
    Class of tools for visualizing the simulation.
    """

    def __init__(
        self, customer_list, starting_time, background, mask, mask_exit, scale=0.2
    ):
        """    
        Parameters
        ----------
        customer_list : list
            List of customer objects
        starting_time : Datetime.time
            Starting time of the simulation
        background: image
            Supermarket background
        mask: image
            Supermarket mask for pathfinding
        mask_exit: image
            Supermarket mask for pathfinding when heading to checkout
        """
        self.customer_list = customer_list
        self.starting_time = starting_time
        self.background = background

        # resize mask for a more efficient Astar pathfinding
        self.mask = cv2.resize(
            mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.threshold(self.mask, 1, 1, cv2.THRESH_BINARY)[1]

        self.mask_exit = cv2.resize(
            mask_exit, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        self.mask_exit = cv2.cvtColor(self.mask_exit, cv2.COLOR_BGR2GRAY)
        self.mask_exit = cv2.threshold(self.mask_exit, 1, 1, cv2.THRESH_BINARY)[1]

        self.scale = scale

    def visualize(self):
        """Start visualization"""
        i = 0
        self.current_time = self.starting_time

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (1, 1, 1)
        lineType = 2

        while True:
            layer = np.zeros((self.background.shape[0], self.background.shape[1], 3))
            frame = self.background.copy()
            if i % 200 == 0:
                self.current_time = add_minute(self.starting_time, i / 200)
                for self.customer_ID, customer in enumerate(self.customer_list):
                    customer.move(
                        self.current_time, self.mask, self.mask_exit, self.scale
                    )
            for self.customer_ID, customer in enumerate(self.customer_list):
                layer = customer.draw(self.current_time, layer)

            cv2.putText(
                layer,
                self.current_time.strftime("%H:%M:%S"),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
            # filter to only display pixels that are not black
            cnd = layer[:] > 0
            frame[cnd] = layer[cnd]

            cv2.imshow("frame", frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()